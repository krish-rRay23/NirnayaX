"""Tests for Jira config, contracts, and adapters."""

from __future__ import annotations

import json as json_lib
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from nirnayax.jira.adapter import MCPJiraAdapter, MockJiraAdapter
from nirnayax.jira.config import JiraConfig
from nirnayax.jira.types import JiraIssueStatus


def test_jira_config_secret_masking() -> None:
    config = JiraConfig(api_token=SecretStr("secret-token-value-12345"))
    masked = config.masked_token()
    assert "secret" not in masked
    assert "se***45" in masked or "***" in masked
    assert "secret-token-value-12345" not in config.render()


def test_jira_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_URL", "https://jira.test.org")
    monkeypatch.setenv("JIRA_USER", "test-user@test.org")
    monkeypatch.setenv("JIRA_API_TOKEN", "my-secret-key")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "TEST")
    monkeypatch.setenv("JIRA_ALLOW_REST_FALLBACK", "false")

    config = JiraConfig.from_env()
    assert config.jira_url == "https://jira.test.org"
    assert config.user_email == "test-user@test.org"
    assert config.project_key == "TEST"
    assert config.api_token.get_secret_value() == "my-secret-key"
    assert config.allow_rest_fallback is False


def test_mock_jira_adapter_lifecycle() -> None:
    adapter = MockJiraAdapter(initial_counter=500)
    issue = adapter.create_issue(
        summary="BGP session down",
        description="Neighbor state active",
        priority="P1",
        labels=("network", "bgp"),
    )

    assert issue.key == "INC-501"
    assert issue.status == JiraIssueStatus.OPEN.value
    assert issue.priority == "P1"

    # Comment
    adapter.add_comment(issue.key, "Triage complete. Predicted BGP_ROUTING.")
    issue_after_comment = adapter.get_issue(issue.key)
    assert issue_after_comment is not None
    assert len(issue_after_comment.comments) == 1
    assert "Triage complete" in issue_after_comment.comments[0]

    # Transition
    adapter.transition_issue(issue.key, JiraIssueStatus.RESOLVED.value)
    issue_resolved = adapter.get_issue(issue.key)
    assert issue_resolved is not None
    assert issue_resolved.status == JiraIssueStatus.RESOLVED.value

    # Action records audit
    records = adapter.action_records
    assert len(records) >= 3
    assert records[0].action_type == "CREATE_ISSUE"
    assert records[1].action_type == "ADD_COMMENT"
    assert records[2].action_type == "TRANSITION"


def test_mcp_jira_adapter_delegation() -> None:
    mock = MockJiraAdapter()
    mcp_adapter = MCPJiraAdapter(mock)

    issue = mcp_adapter.create_issue("Summary", "Description")
    assert issue.key.startswith("INC-")

    mcp_adapter.add_comment(issue.key, "MCP comment")
    mcp_adapter.transition_issue(issue.key, "In Progress")

    retrieved = mcp_adapter.get_issue(issue.key)
    assert retrieved is not None
    assert retrieved.status == "In Progress"
    assert len(retrieved.comments) == 1


class RovoMCPMockHTTPClient:
    """Mock HTTP Client simulating official Rovo MCP V2 JSON-RPC 2.0 protocol."""

    def __init__(self, response_factory: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response_factory = response_factory

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        headers_dict = headers or {}
        json_body = json or {}
        self.calls.append({"url": url, "json": json_body, "headers": headers_dict})
        if self.response_factory:
            res: httpx.Response = self.response_factory(url, json_body, headers_dict)
            return res

        method = json_body.get("method", "")
        tool = json_body.get("params", {}).get("name", "")
        args = json_body.get("params", {}).get("arguments", {})

        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "sess-rovo-test-123"},
                json={
                    "jsonrpc": "2.0",
                    "id": json_body.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "atlassian-mcp-server", "version": "1.0.0"},
                    },
                },
            )

        if method == "notifications/initialized":
            return httpx.Response(200, json={"jsonrpc": "2.0"})

        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json_body.get("id"),
                    "result": {
                        "tools": [
                            {"name": "createJiraIssue"},
                            {"name": "getJiraIssue"},
                            {"name": "addOrEditJiraIssueComment"},
                            {"name": "transitionJiraIssue"},
                            {"name": "editJiraIssue"},
                            {"name": "executeRead"},
                            {"name": "executeWrite"},
                        ]
                    },
                },
            )

        # Tool calls
        if tool in ("createJiraIssue", "executeWrite"):
            inner_res = json_lib.dumps({"data": {"id": "10099", "key": "ACME-777"}})
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json_body.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": inner_res}],
                        "isError": False,
                    },
                },
            )
        elif tool in ("getJiraIssue", "executeRead"):
            key_val = args.get("issueIdOrKey") or args.get("inputs", {}).get(
                "issueIdOrKey", "ACME-777"
            )
            inner_res = json_lib.dumps({
                "data": {
                    "id": "10099",
                    "key": key_val,
                    "fields": {
                        "summary": "Retrieved issue via Rovo MCP V2",
                        "description": "Details",
                        "status": {"name": "In Progress"},
                        "priority": {"name": "P1"},
                    },
                }
            })
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json_body.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": inner_res}],
                        "isError": False,
                    },
                },
            )
        else:
            inner_res = json_lib.dumps({"data": {"status": "ok", "updated": True}})
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json_body.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": inner_res}],
                        "isError": False,
                    },
                },
            )


def test_mcp_jira_adapter_rovo_v2_full_protocol() -> None:
    config = JiraConfig(
        jira_url="https://acme.atlassian.net",
        user_email="admin@acme.com",
        api_token=SecretStr("token-12345"),
        project_key="ACME",
        mcp_endpoint="https://mcp.atlassian.com/v2/mcp",
        allow_rest_fallback=False,
    )
    mock_http = RovoMCPMockHTTPClient()
    adapter = MCPJiraAdapter(config=config, http_client=mock_http)

    # 1. Discover tools
    discovered = adapter.discover_tools()
    assert "createJiraIssue" in discovered
    assert "getJiraIssue" in discovered

    # 2. create_issue
    issue = adapter.create_issue("BGP session down", "Neighbor active", priority="P1")
    assert issue.key == "ACME-777"
    assert issue.priority == "P1"

    # 3. get_issue
    retrieved = adapter.get_issue("ACME-777")
    assert retrieved is not None
    assert retrieved.key == "ACME-777"
    assert retrieved.summary == "Retrieved issue via Rovo MCP V2"

    # 4. add_comment
    adapter.add_comment("ACME-777", "Investigation completed")

    # 5. transition_issue
    adapter.transition_issue("ACME-777", "Resolved")

    # 6. update_issue
    adapter.update_issue("ACME-777", {"assignee": "oncall"})

    # Verify session & headers
    assert len(mock_http.calls) >= 5
    first_call = mock_http.calls[0]

    assert first_call["url"] == "https://mcp.atlassian.com/v2/mcp"
    assert first_call["headers"]["Content-Type"] == "application/json"
    assert "Basic " in first_call["headers"]["Authorization"]
    assert first_call["headers"]["X-Atlassian-Base-Url"] == "https://acme.atlassian.net"

    # Verify subsequent calls include Mcp-Session-Id header
    subsequent_call = mock_http.calls[2]
    assert subsequent_call["headers"].get("Mcp-Session-Id") == "sess-rovo-test-123"

    # Verify audit records
    records = adapter.action_records
    assert len(records) == 4
    assert records[0].action_type == "CREATE_ISSUE"
    assert records[1].action_type == "ADD_COMMENT"
    assert records[2].action_type == "TRANSITION"
    assert records[3].action_type == "UPDATE_ISSUE"


def test_mcp_jira_adapter_no_silent_rest_fallback() -> None:
    config = JiraConfig(
        mcp_endpoint="https://mcp.atlassian.com/v2/mcp",
        allow_rest_fallback=False,
    )

    def error_factory(
        url: str, json_body: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        return httpx.Response(500, text="MCP Error: Service Unavailable")

    mock_http = RovoMCPMockHTTPClient(response_factory=error_factory)
    adapter = MCPJiraAdapter(config=config, http_client=mock_http)

    with pytest.raises(RuntimeError) as exc_info:
        adapter.create_issue("Title", "Description")

    err_str = str(exc_info.value)
    assert "500" in err_str or "initialize failed" in err_str or "HTTP" in err_str
