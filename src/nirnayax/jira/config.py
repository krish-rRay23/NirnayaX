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
    enabled: bool = True

    @classmethod
    def from_env(cls) -> JiraConfig:
        """Load configuration from environment variables safely."""

        url = os.getenv("JIRA_URL", "https://jira.company.internal")
        user = os.getenv("JIRA_USER") or os.getenv("JIRA_EMAIL") or "nirnayax-bot@company.internal"
        token = os.getenv("JIRA_API_TOKEN", "mock-jira-api-token")
        project = os.getenv("JIRA_PROJECT_KEY", "INC")
        enabled_str = os.getenv("JIRA_ENABLED", "true").lower()
        enabled = enabled_str in ("true", "1", "yes")

        return cls(
            jira_url=url,
            user_email=user,
            api_token=SecretStr(token),
            project_key=project,
            enabled=enabled,
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
            f"project={self.project_key}, token={self.masked_token()})"
        )


__all__ = ["JiraConfig"]
