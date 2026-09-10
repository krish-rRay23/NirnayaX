"""Configuration and secret handling for Jira integration."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, SecretStr


class JiraConfig(BaseModel):
    """Jira connection settings loaded safely from environment variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    jira_url: str = "https://jira.company.internal"
    user_email: str = "nirnayax-bot@company.internal"
    api_token: SecretStr = SecretStr("mock-jira-api-token")
    project_key: str = "INC"
    mcp_endpoint: str = "https://mcp.atlassian.com/v2/mcp"
    enabled: bool = True
    allow_rest_fallback: bool = False

    @classmethod
    def from_env(cls) -> JiraConfig:
        """Load configuration from environment variables safely."""

        url = (
            os.getenv("JIRA_BASE_URL")
            or os.getenv("JIRA_URL")
            or "https://jira.company.internal"
        )
        user = (
            os.getenv("JIRA_EMAIL")
            or os.getenv("JIRA_USER")
            or "nirnayax-bot@company.internal"
        )
        token = os.getenv("JIRA_API_TOKEN", "mock-jira-api-token")
        project = os.getenv("JIRA_PROJECT_KEY", "INC")
        mcp_endpoint = os.getenv("JIRA_MCP_ENDPOINT", "https://mcp.atlassian.com/v2/mcp")
        enabled_str = os.getenv("JIRA_ENABLED", "true").lower()
        enabled = enabled_str in ("true", "1", "yes")
        fallback_str = os.getenv("JIRA_ALLOW_REST_FALLBACK", "false").lower()
        allow_rest_fallback = fallback_str in ("true", "1", "yes")

        return cls(
            jira_url=url,
            user_email=user,
            api_token=SecretStr(token),
            project_key=project,
            mcp_endpoint=mcp_endpoint,
            enabled=enabled,
            allow_rest_fallback=allow_rest_fallback,
        )

    def masked_token(self) -> str:
        """Return a masked representation of the API token for logs."""

        raw = self.api_token.get_secret_value()
        if len(raw) <= 4:
            return "****"
        return f"{raw[:2]}***{raw[-2:]}"

    def render(self) -> str:
        return (
            f"JiraConfig(url={self.jira_url}, user={self.user_email}, "
            f"project={self.project_key}, endpoint={self.mcp_endpoint}, "
            f"token={self.masked_token()})"
        )


__all__ = ["JiraConfig"]
