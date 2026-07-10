"""mcp_server — CLI 코어 위에 얹는 얇은 MCP 제어 표면(선택).

Claude/dev-hub 에서 대화로 job을 띄우고 상태를 보게 한다. 무인 페일오버 자체는 CLI가
전담하고, 여긴 사람이 붙어있을 때의 편의 계층일 뿐(v1.1 애드온).

실행: pip install 'freecloud[mcp]'  &&  python -m freecloud.mcp_server
등록(예, Claude Code): claude mcp add freecloud -- python -m freecloud.mcp_server
"""
from __future__ import annotations

import json


def _build():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit("pip install 'freecloud[mcp]' 필요.") from e

    from . import registry, state, orchestrator
    from .job import Job
    from .catalog import CATALOG, render

    mcp = FastMCP("freecloud")

    @mcp.tool()
    def list_providers() -> str:
        """등록된 provider와 probe 상태(가용/사유)를 반환."""
        out = []
        for name in registry.available_names():
            pr = registry.get(name).probe()
            out.append({"provider": name, "available": pr.available, "reason": pr.reason})
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def cooldown_status() -> str:
        """provider별 남은 쿨다운(초)."""
        return json.dumps(state.snapshot(), ensure_ascii=False)

    @mcp.tool()
    def clouds() -> str:
        """지원/후보 무료(비중국) 클라우드 카탈로그(텍스트)."""
        return render(CATALOG)

    @mcp.tool()
    def run_job(job_yaml_path: str, once: bool = False) -> str:
        """job.yaml을 페일오버 실행하고 결과 JSON을 반환(블로킹)."""
        result = orchestrator.run(Job.load(job_yaml_path), once=once)
        return json.dumps(result, ensure_ascii=False)

    return mcp


def main() -> None:
    _build().run()


if __name__ == "__main__":
    main()
