"""Tests for Jira config, contracts, and adapters."""

from __future__ import annotations

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

    config = JiraConfig.from_env()
    assert config.jira_url == "https://jira.test.org"
    assert config.user_email == "test-user@test.org"
    assert config.project_key == "TEST"
    assert config.api_token.get_secret_value() == "my-secret-key"


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
