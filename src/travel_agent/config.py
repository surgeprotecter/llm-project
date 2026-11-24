from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model_name: str = "deepseek-flash"
    api_base: str = "https://api.deepseek.com"
    api_key: str | None = None
    database_path: Path = Path("travel_agent.sqlite3")
    context_token_budget: int = 36_000
    max_planner_steps: int = 8
    max_execute_rounds: int = 24
    max_tool_retries: int = 2
    summary_model_name: str = "deepseek-flash"
    output_dir: Path = Path("output")
    mcp_config_path: Path | None = None
    require_real_model: bool = False
    tool_mode: Literal["mock", "real"] = "mock"

    @property
    def strict_real(self) -> bool:
        return self.require_real_model or self.tool_mode == "real"

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file:
            load_dotenv(env_file, override=False)
        tool_mode = os.getenv("TRAVEL_AGENT_TOOL_MODE", "mock").lower()
        return cls(
            model_name=os.getenv("TRAVEL_AGENT_MODEL", "deepseek-flash"),
            api_base=os.getenv("TRAVEL_AGENT_API_BASE", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            database_path=Path(os.getenv("TRAVEL_AGENT_DATABASE_PATH", "travel_agent.sqlite3")),
            context_token_budget=int(os.getenv("TRAVEL_AGENT_CONTEXT_TOKEN_BUDGET", "36000")),
            max_planner_steps=int(os.getenv("TRAVEL_AGENT_MAX_PLANNER_STEPS", "8")),
            max_execute_rounds=int(os.getenv("TRAVEL_AGENT_MAX_EXECUTE_ROUNDS", "24")),
            max_tool_retries=int(os.getenv("TRAVEL_AGENT_MAX_TOOL_RETRIES", "2")),
            summary_model_name=os.getenv("TRAVEL_AGENT_SUMMARY_MODEL", "deepseek-flash"),
            output_dir=Path(os.getenv("TRAVEL_AGENT_OUTPUT_DIR", "output")),
            mcp_config_path=Path(os.getenv("TRAVEL_AGENT_MCP_CONFIG")) if os.getenv("TRAVEL_AGENT_MCP_CONFIG") else None,
            require_real_model=(
                os.getenv("TRAVEL_AGENT_REQUIRE_REAL_MODEL", "false").lower() in {"1", "true", "yes"}
                or tool_mode == "real"
            ),
            tool_mode=tool_mode,
        )
