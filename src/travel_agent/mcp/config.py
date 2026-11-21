from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MCPServerConfig(BaseModel):
    """Fixed, application-owned configuration for one MCP server."""

    name: str
    transport: Literal["stdio", "streamable_http"] = "stdio"
    enabled: bool = True
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    optional_headers: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 1

    @model_validator(mode="after")
    def validate_transport_settings(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server requires command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http MCP server requires url")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return self


def load_mcp_configs(path: str | Path | None) -> list[MCPServerConfig]:
    if not path:
        return []
    config_path = Path(path)
    if not config_path.exists():
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_servers = payload.get("servers", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_servers, list):
        raise ValueError("MCP config must be a list or an object with a servers list")
    return [MCPServerConfig.model_validate(item) for item in raw_servers]
