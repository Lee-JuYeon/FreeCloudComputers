"""catalog — 섹션 구조 불변식 + render 레이아웃 + Lab 블록.

특정 항목 이름을 박지 않고(값은 수시 변동) 구조 불변식으로 검증한다:
  · 모든 Entry.section ∈ SECTIONS 키 / status ∈ {adapter,candidate} / credit_card ∈ {no,yes,gcp}
  · render 출력에 섹션 헤더가 SECTIONS 순서로 나오고, 각 항목이 자기 헤더와 다음 헤더 사이에 위치
  · 빈 섹션(mobile)은 헤더 + 안내줄
  · Lab 블록이 LABS 항목과 함께 출력에 포함
  · _cmd_clouds 가 rc 0 + 전 섹션 헤더를 찍는다
"""
from __future__ import annotations

import sys
import types

import pytest

from freecloud import cli
from freecloud.catalog import CATALOG, LABS, SECTIONS, Entry, Lab, render


SECTION_KEYS = {key for key, _ in SECTIONS}


def test_entry_fields_are_valid():
    assert CATALOG, "카탈로그가 비어 있으면 안 된다"
    for e in CATALOG:
        assert isinstance(e, Entry)
        assert e.section in SECTION_KEYS
        assert e.status in {"adapter", "candidate"}        # control 폐지
        assert e.credit_card in {"no", "yes", "gcp"}


def test_sections_are_unique_and_ordered():
    keys = [key for key, _ in SECTIONS]
    assert len(keys) == len(set(keys)), "섹션 키 중복 금지"
    assert keys == ["gpu", "computer", "mobile"], "섹션 순서(정본)를 고정한다"


def test_mobile_section_is_empty():
    assert "mobile" in SECTION_KEYS
    assert not [e for e in CATALOG if e.section == "mobile"]


def test_render_section_headers_appear_in_sections_order():
    out = render(CATALOG)
    positions = []
    for _, header in SECTIONS:
        idx = out.find(header)
        assert idx != -1, f"섹션 헤더 누락: {header}"
        positions.append(idx)
    assert positions == sorted(positions), "섹션 헤더가 SECTIONS 순서대로 나와야 한다"


def test_each_entry_sits_under_its_own_section_header():
    out = render(CATALOG)
    # 헤더별 (시작, 끝) 범위 — 다음 헤더(또는 Lab 블록) 전까지
    header_idx = {key: out.find(h) for key, h in SECTIONS}
    lab_idx = out.find("Lab — 시뮬 런타임")
    assert lab_idx != -1
    ordered = sorted(SECTIONS, key=lambda kh: header_idx[kh[0]])
    for i, (key, _) in enumerate(ordered):
        start = header_idx[key]
        end = header_idx[ordered[i + 1][0]] if i + 1 < len(ordered) else lab_idx
        for e in CATALOG:
            if e.section != key:
                continue
            pos = out.find(e.name)
            assert start < pos < end, f"{e.name} 이 {key} 섹션 밖에 있다"


def test_empty_section_shows_header_and_notice():
    out = render(CATALOG)
    lines = out.splitlines()
    # mobile 헤더 바로 뒤에 '등록 항목 없음' 안내줄
    _, mobile_header = next(kh for kh in SECTIONS if kh[0] == "mobile")
    hi = next(i for i, ln in enumerate(lines) if mobile_header in ln)
    assert any("등록 항목 없음" in ln for ln in lines[hi + 1:hi + 3])


def test_lab_block_present_with_all_labs():
    assert LABS and all(isinstance(x, Lab) for x in LABS)
    out = render(CATALOG)
    assert "Lab — 시뮬 런타임" in out
    for lab in LABS:
        assert lab.name in out
        assert lab.requires in out


def test_lab_free_pool_decision_is_fixed():
    # 사용자 결정 고정: MJX+Brax=무료풀 구동 가능, Isaac Lab=불가
    assert {l.name: l.free_pool for l in LABS} == {"MJX+Brax": True, "Isaac Lab": False}


def test_render_lab_line_shows_matching_pool_label():
    lines = render(CATALOG).splitlines()
    for lab in LABS:
        line = next(ln for ln in lines if lab.name in ln)
        # note 문구에 "불가"가 섞여 있어 줄 전체를 보면 반쪽 검사가 된다 → 라벨 칸(note 앞)만 보고 양방향 대칭 검사.
        head = line.split(lab.note)[0]
        want, other = ("가능", "불가") if lab.free_pool else ("불가", "가능")
        assert want in head and other not in head


def test_render_rejects_entry_with_unknown_section():
    bad = CATALOG + [Entry("quantum", "X", "qbit", "무제한", "no", "candidate", "")]
    with pytest.raises(ValueError):
        render(bad)


def test_cmd_clouds_returns_zero_and_prints_all_headers(capsys):
    rc = cli.main(["clouds"])
    out = capsys.readouterr().out
    assert rc == 0
    for _, header in SECTIONS:
        assert header in out
    assert "Lab — 시뮬 런타임" in out


def test_mcp_clouds_tool_returns_render(monkeypatch):
    """mcp_server 의 clouds 도구가 render(CATALOG) 를 돌려주는지 — FastMCP 를 스텁해서 확인.

    이 환경엔 mcp.server.fastmcp 가 없으므로(실설치 금지) 가짜 모듈을 심는다.
    tool() 데코레이터는 함수를 이름으로 모아, 테스트가 clouds 를 직접 호출할 수 있게 한다.
    """
    class FakeMCP:
        def __init__(self, name):
            self.tools = {}

        def tool(self):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mod = types.ModuleType("mcp.server.fastmcp")
    mod.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mod)

    from freecloud import mcp_server
    mcp = mcp_server._build()
    assert mcp.tools["clouds"]() == render(CATALOG)
