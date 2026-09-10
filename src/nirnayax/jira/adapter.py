"""Jira integration adapters: provider protocol, in-memory mock, and MCP wrapper."""

from __future__ import annotations

import base64
import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import JiraConfig
from .types import JiraActionRecord, JiraIssue, JiraIssueStatus


@runtime_checkable
class JiraAdapter(Protocol):
    """Clean provider boundary for Jira incident ticket operations."""

    def create_issue(
        self,
        summary: str,
        description: str,
        priority: str = "P2",
        labels: tuple[str, ...] = (),
    ) -> JiraIssue: ...

    def add_comment(self, issue_key: str, comment: str) -> None: ...

    def transition_issue(self, issue_key: str, status: str) -> None: ...

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None: ...

    def get_issue(self, issue_key: str) -> JiraIssue | None: ...

    @property
    def action_records(self) -> tuple[JiraActionRecord, ...]: ...


class MockJiraAdapter:
    """Deterministic in-memory Jira adapter for offline testing and CLI demos."""

    def __init__(self, config: JiraConfig | None = None, initial_counter: int = 1000) -> None:
        self.config = config or JiraConfig()
        self._counter = initial_counter
        self._issues: dict[str, JiraIssue] = {}
        self._records: list[JiraActionRecord] = []

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _record(self, action_type: str, issue_key: str, details: str) -> None:
        rec = JiraActionRecord(
            action_type=action_type,
            issue_key=issue_key,
            timestamp=self._now(),
            details=details,
        )
        self._records.append(rec)

    @property
    def action_records(self) -> tuple[JiraActionRecord, ...]:
        return tuple(self._records)

    def create_issue(
        self,
        summary: str,
        description: str,
        priority: str = "P2",
        labels: tuple[str, ...] = (),
    ) -> JiraIssue:
        self._counter += 1
        key = f"{self.config.project_key}-{self._counter}"
        now = self._now()
        issue = JiraIssue(
            key=key,
            summary=summary,
            description=description,
            issue_type="Incident",
            status=JiraIssueStatus.OPEN.value,
            priority=priority,
            assignee=None,
            comments=(),
            labels=tuple(labels),
            created_at=now,
            updated_at=now,
        )
        self._issues[key] = issue
        self._record("CREATE_ISSUE", key, f"Created issue '{summary}' (priority={priority})")
        return issue

    def add_comment(self, issue_key: str, comment: str) -> None:
        issue = self._issues.get(issue_key)
        if issue is None:
            # Create a placeholder issue if key was externally provided
            now = self._now()
            issue = JiraIssue(
                key=issue_key,
                summary=f"External issue {issue_key}",
                description="Placeholder issue",
                created_at=now,
                updated_at=now,
            )

        updated_comments = (*issue.comments, comment)
        data = issue.model_dump()
        data["comments"] = updated_comments
        data["updated_at"] = self._now()
        self._issues[issue_key] = JiraIssue(**data)
        short = comment[:60].replace("\n", " ")
        self._record("ADD_COMMENT", issue_key, f"Added comment: '{short}...'")

    def transition_issue(self, issue_key: str, status: str) -> None:
        issue = self._issues.get(issue_key)
        if issue is not None:
            data = issue.model_dump()
            data["status"] = status
            data["updated_at"] = self._now()
            self._issues[issue_key] = JiraIssue(**data)
        self._record("TRANSITION", issue_key, f"Status transitioned to '{status}'")

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        issue = self._issues.get(issue_key)
        if issue is not None:
            data = issue.model_dump()
            data.update(fields)
            data["updated_at"] = self._now()
            self._issues[issue_key] = JiraIssue(**data)
        self._record("UPDATE_ISSUE", issue_key, f"Updated fields: {list(fields.keys())}")

    def get_issue(self, issue_key: str) -> JiraIssue | None:
        return self._issues.get(issue_key)


class MCPJiraAdapter:
    """Official Atlassian Rovo MCP v2 protocol integration for Jira Cloud operations."""

    def __init__(
        self,
        inner: JiraAdapter | None = None,
        config: JiraConfig | None = None,
        *,
        http_client: Any | None = None,
    ) -> None:
        self._inner = inner
        self.config = config or JiraConfig.from_env()
        self._http_client = http_client
        self._records: list[JiraActionRecord] = []
        self._session_id: str | None = None
        self._discovered_tools: list[str] | None = None

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _record(self, action_type: str, issue_key: str, details: str) -> None:
        rec = JiraActionRecord(
            action_type=action_type,
            issue_key=issue_key,
            timestamp=self._now(),
            details=details,
        )
        self._records.append(rec)

    @property
    def action_records(self) -> tuple[JiraActionRecord, ...]:
        if self._inner is not None:
            return self._inner.action_records
        return tuple(self._records)

    def _headers(self) -> dict[str, str]:
        auth_str = f"{self.config.user_email}:{self.config.api_token.get_secret_value()}"
        b64_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Basic {b64_auth}",
            "X-Atlassian-Base-Url": self.config.jira_url,
            "X-Atlassian-Project": self.config.project_key,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _parse_mcp_response(self, text: str) -> dict[str, Any]:
        """Parse SSE formatted string lines or raw JSON-RPC 2.0 response."""
        for line in text.splitlines():
            if line.startswith("data:"):
                d_str = line[5:].strip()
                if d_str:
                    try:
                        res: dict[str, Any] = json.loads(d_str)
                        return res
                    except Exception:
                        pass
        try:
            r: dict[str, Any] = json.loads(text)
            return r
        except Exception:
            return {}

    def _ensure_mcp_session(self, client: Any) -> None:
        """Establish session with Rovo MCP V2 server using JSON-RPC initialize."""
        if self._session_id is not None:
            return

        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "NirnayaX", "version": "0.1.0"},
            },
        }
        res = client.post(self.config.mcp_endpoint, json=init_payload, headers=self._headers())
        if res.status_code not in (200, 201):
            if not self.config.allow_rest_fallback:
                raise RuntimeError(
                    f"Rovo MCP V2 initialize failed with HTTP {res.status_code}"
                )
            return

        sid = res.headers.get("mcp-session-id") or res.headers.get("Mcp-Session-Id")
        data = self._parse_mcp_response(res.text)
        if not sid and isinstance(data.get("result"), dict):
            sid = data["result"].get("sessionId")
        if sid:
            self._session_id = sid

        notif_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        with contextlib.suppress(Exception):
            client.post(self.config.mcp_endpoint, json=notif_payload, headers=self._headers())

    def discover_tools(self) -> list[str]:
        """Discover tools registered on the Atlassian Rovo MCP V2 server."""
        if self._discovered_tools is not None:
            return self._discovered_tools

        payload = {
            "jsonrpc": "2.0",
            "id": f"mcp-disc-{uuid.uuid4().hex[:6]}",
            "method": "tools/list",
            "params": {},
        }
        try:
            if self._http_client is not None:
                self._ensure_mcp_session(self._http_client)
                res = self._http_client.post(
                    self.config.mcp_endpoint, json=payload, headers=self._headers()
                )
            else:
                with httpx.Client(timeout=15.0) as client:
                    self._ensure_mcp_session(client)
                    res = client.post(
                        self.config.mcp_endpoint, json=payload, headers=self._headers()
                    )
            if res.status_code in (200, 201):
                data = self._parse_mcp_response(res.text)
                tools = data.get("result", {}).get("tools", [])
                tnames: list[str] = [
                    str(t["name"])
                    for t in tools
                    if isinstance(t, dict) and t.get("name") is not None
                ]
                self._discovered_tools = tnames
                return tnames
        except Exception:
            pass

        self._discovered_tools = []
        return []

    def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send JSON-RPC 2.0 request over Rovo MCP v2 protocol."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"mcp-req-{uuid.uuid4().hex[:8]}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        if self._http_client is not None:
            self._ensure_mcp_session(self._http_client)
            res = self._http_client.post(
                self.config.mcp_endpoint, json=payload, headers=self._headers()
            )
        else:
            with httpx.Client(timeout=15.0) as client:
                self._ensure_mcp_session(client)
                res = client.post(
                    self.config.mcp_endpoint, json=payload, headers=self._headers()
                )

        if res.status_code not in (200, 201):
            if not self.config.allow_rest_fallback:
                raise RuntimeError(
                    f"Rovo MCP endpoint returned HTTP status {res.status_code}"
                )
            return {}

        data = self._parse_mcp_response(res.text)
        if "error" in data:
            if not self.config.allow_rest_fallback:
                raise RuntimeError(f"Rovo MCP tool '{tool_name}' error: {data['error']}")
            return {}

        result = data.get("result", {})
        if isinstance(result, dict):
            if result.get("isError"):
                content = result.get("content", [])
                err_text = (
                    content[0].get("text")
                    if content and isinstance(content, list) and isinstance(content[0], dict)
                    else "Unknown Rovo MCP error"
                )
                if not self.config.allow_rest_fallback:
                    raise RuntimeError(f"Rovo MCP tool '{tool_name}' error: {err_text}")
                return {}

            content = result.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                item = content[0]
                if isinstance(item, dict) and "text" in item:
                    txt = item["text"]
                    if isinstance(txt, str) and txt.strip():
                        try:
                            inner_json = json.loads(txt)
                            if isinstance(inner_json, dict):
                                data_obj = inner_json.get("data", inner_json)
                                if isinstance(data_obj, dict):
                                    return data_obj
                        except Exception:
                            pass
            return result
        return {}

    def create_issue(
        self,
        summary: str,
        description: str,
        priority: str = "P2",
        labels: tuple[str, ...] = (),
    ) -> JiraIssue:
        if self._inner is not None:
            return self._inner.create_issue(summary, description, priority, labels)

        issue_key = None
        priority_map = {"P1": "High", "P2": "Medium", "P3": "Low", "P4": "Lowest"}
        jira_priority = priority_map.get(priority, priority)

        try:
            # 1. Try createJiraIssue directly via Rovo MCP V2
            args: dict[str, Any] = {
                "cloudId": self.config.jira_url,
                "projectKey": self.config.project_key,
                "summary": summary,
                "description": description,
                "issueType": "Task",
                "priority": jira_priority,
            }
            if labels:
                args["labels"] = list(labels)

            resp = self._call_mcp_tool("createJiraIssue", args)
            raw_key = resp.get("key") or resp.get("issueKey")

            # 2. Try executeWrite wrapper if direct tool name didn't return key directly
            if not raw_key:
                exec_args = {
                    "cloudId": self.config.jira_url,
                    "name": "createJiraIssue",
                    "inputs": {
                        "projectKey": self.config.project_key,
                        "summary": summary,
                        "description": description,
                        "issueType": "Task",
                        "priority": priority,
                        "labels": list(labels),
                    },
                }
                resp = self._call_mcp_tool("executeWrite", exec_args)
                raw_key = resp.get("key") or resp.get("issueKey")

            if raw_key:
                issue_key = str(raw_key)

        except Exception as err:
            if not self.config.allow_rest_fallback:
                raise RuntimeError(f"Failed to create issue via Rovo MCP V2: {err}") from err

        if not issue_key:
            if self.config.allow_rest_fallback:
                issue_key = self._rest_create_issue(summary, description, priority, labels)
            else:
                raise RuntimeError("Rovo MCP V2 createJiraIssue failed to return created issue key")

        now = self._now()
        issue = JiraIssue(
            key=issue_key,
            summary=summary,
            description=description,
            issue_type="Task",
            status=JiraIssueStatus.OPEN.value,
            priority=priority,
            assignee=None,
            comments=(),
            labels=tuple(labels),
            created_at=now,
            updated_at=now,
        )
        self._record("CREATE_ISSUE", issue_key, f"Created issue '{summary}' via Rovo MCP V2")
        return issue

    def _rest_create_issue(
        self, summary: str, description: str, priority: str, labels: tuple[str, ...]
    ) -> str:
        url = f"{self.config.jira_url.rstrip('/')}/rest/api/3/issue"
        headers = self._headers()
        headers["Accept"] = "application/json"

        body = {
            "fields": {
                "project": {"key": self.config.project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
            }
        }

        if self._http_client is not None:
            res = self._http_client.post(url, json=body, headers=headers)
        else:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(url, json=body, headers=headers)

        if res.status_code not in (200, 201):
            raise RuntimeError(f"Jira REST API create failed ({res.status_code}): {res.text}")

        data = res.json()
        k: str = data.get("key", f"{self.config.project_key}-1001")
        return k

    def add_comment(self, issue_key: str, comment: str) -> None:
        if self._inner is not None:
            self._inner.add_comment(issue_key, comment)
            return

        try:
            args = {
                "cloudId": self.config.jira_url,
                "issueIdOrKey": issue_key,
                "comment": comment,
            }
            self._call_mcp_tool("addOrEditJiraIssueComment", args)
        except Exception as err:
            if not self.config.allow_rest_fallback:
                raise RuntimeError(f"Failed to add comment via Rovo MCP V2: {err}") from err
            self._rest_add_comment(issue_key, comment)

        short = comment[:60].replace("\n", " ")
        self._record("ADD_COMMENT", issue_key, f"Comment via Rovo MCP V2: '{short}...'")

    def _rest_add_comment(self, issue_key: str, comment: str) -> None:
        url = f"{self.config.jira_url.rstrip('/')}/rest/api/3/issue/{issue_key}/comment"
        headers = self._headers()
        headers["Accept"] = "application/json"
        body = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": comment}],
                    }
                ],
            }
        }
        if self._http_client is not None:
            res = self._http_client.post(url, json=body, headers=headers)
        else:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(url, json=body, headers=headers)

        if res.status_code not in (200, 201):
            raise RuntimeError(f"Jira REST API add comment failed ({res.status_code}): {res.text}")

    def transition_issue(self, issue_key: str, status: str) -> None:
        if self._inner is not None:
            self._inner.transition_issue(issue_key, status)
            return

        try:
            args = {
                "cloudId": self.config.jira_url,
                "issueIdOrKey": issue_key,
                "status": status,
            }
            self._call_mcp_tool("transitionJiraIssue", args)
        except Exception as err:
            if not self.config.allow_rest_fallback:
                raise RuntimeError(f"Failed to transition issue via Rovo MCP V2: {err}") from err

        self._record("TRANSITION", issue_key, f"Status '{status}' via Rovo MCP V2")

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        if self._inner is not None:
            self._inner.update_issue(issue_key, fields)
            return

        try:
            args = {
                "cloudId": self.config.jira_url,
                "issueIdOrKey": issue_key,
                "fields": fields,
            }
            self._call_mcp_tool("editJiraIssue", args)
        except Exception as err:
            if not self.config.allow_rest_fallback:
                raise RuntimeError(f"Failed to update issue via Rovo MCP V2: {err}") from err

        f_list = list(fields.keys())
        self._record("UPDATE_ISSUE", issue_key, f"Updated {f_list} via Rovo MCP V2")

    def get_issue(self, issue_key: str) -> JiraIssue | None:
        if self._inner is not None:
            return self._inner.get_issue(issue_key)

        try:
            args = {"cloudId": self.config.jira_url, "issueIdOrKey": issue_key}
            resp = self._call_mcp_tool("getJiraIssue", args)

            if not resp:
                exec_args = {
                    "cloudId": self.config.jira_url,
                    "name": "getJiraIssue",
                    "inputs": {"issueIdOrKey": issue_key},
                }
                resp = self._call_mcp_tool("executeRead", exec_args)

            if resp:
                fields = resp.get("fields", {}) if isinstance(resp.get("fields"), dict) else resp
                raw_summary = fields.get("summary") or resp.get("summary", f"Issue {issue_key}")
                summary_val = str(raw_summary)
                desc = resp.get("description", "Retrieved via Rovo MCP V2")
                desc_val = str(fields.get("description") or desc)
                status_obj = fields.get("status")
                status_val = str(
                    status_obj.get("name")
                    if isinstance(status_obj, dict)
                    else resp.get("status", JiraIssueStatus.OPEN.value)
                )
                prio_obj = fields.get("priority")
                priority_val = str(
                    prio_obj.get("name")
                    if isinstance(prio_obj, dict)
                    else resp.get("priority", "P2")
                )

                now = self._now()
                return JiraIssue(
                    key=str(resp.get("key", issue_key)),
                    summary=summary_val,
                    description=desc_val,
                    issue_type=str(resp.get("issue_type", "Task")),
                    status=status_val,
                    priority=priority_val,
                    assignee=resp.get("assignee"),
                    comments=tuple(resp.get("comments", ())),
                    labels=tuple(resp.get("labels", ())),
                    created_at=now,
                    updated_at=now,
                )
        except Exception as err:
            if not self.config.allow_rest_fallback:
                msg = f"Failed to get issue '{issue_key}' via Rovo MCP V2: {err}"
                raise RuntimeError(msg) from err

        if self.config.allow_rest_fallback:
            return self._rest_get_issue(issue_key)
        return None

    def _rest_get_issue(self, issue_key: str) -> JiraIssue | None:
        url = f"{self.config.jira_url.rstrip('/')}/rest/api/3/issue/{issue_key}"
        headers = self._headers()
        headers["Accept"] = "application/json"

        if self._http_client is not None:
            res = self._http_client.get(url, headers=headers)
        else:
            with httpx.Client(timeout=15.0) as client:
                res = client.get(url, headers=headers)

        if res.status_code != 200:
            return None

        data = res.json()
        fields = data.get("fields", {})
        now = self._now()
        assignee_data = fields.get("assignee")
        assignee_name = assignee_data.get("displayName") if assignee_data else None

        return JiraIssue(
            key=data.get("key", issue_key),
            summary=fields.get("summary", f"Issue {issue_key}"),
            description="Retrieved via Jira Cloud REST API",
            issue_type=fields.get("issuetype", {}).get("name", "Task"),
            status=fields.get("status", {}).get("name", JiraIssueStatus.OPEN.value),
            priority=fields.get("priority", {}).get("name", "P2"),
            assignee=assignee_name,
            comments=(),
            labels=tuple(fields.get("labels", ())),
            created_at=now,
            updated_at=now,
        )


__all__ = [
    "JiraAdapter",
    "MCPJiraAdapter",
    "MockJiraAdapter",
]
