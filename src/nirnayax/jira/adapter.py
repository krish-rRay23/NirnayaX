"""Jira integration adapters: provider protocol, in-memory mock, and MCP wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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
    """MCP-compatible wrapper delegating to MCP tools or fallback MockJiraAdapter."""

    def __init__(self, inner: JiraAdapter | None = None, config: JiraConfig | None = None) -> None:
        self._inner = inner or MockJiraAdapter(config)
        self._config = config or JiraConfig()

    @property
    def action_records(self) -> tuple[JiraActionRecord, ...]:
        return self._inner.action_records

    def create_issue(
        self,
        summary: str,
        description: str,
        priority: str = "P2",
        labels: tuple[str, ...] = (),
    ) -> JiraIssue:
        return self._inner.create_issue(summary, description, priority, labels)

    def add_comment(self, issue_key: str, comment: str) -> None:
        self._inner.add_comment(issue_key, comment)

    def transition_issue(self, issue_key: str, status: str) -> None:
        self._inner.transition_issue(issue_key, status)

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        self._inner.update_issue(issue_key, fields)

    def get_issue(self, issue_key: str) -> JiraIssue | None:
        return self._inner.get_issue(issue_key)


__all__ = [
    "JiraAdapter",
    "MCPJiraAdapter",
    "MockJiraAdapter",
]
