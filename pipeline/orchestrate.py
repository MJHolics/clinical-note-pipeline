"""LangGraph 오케스트레이션 — ASR→추출(NER)→관계추출(RE)→검증을 StateGraph로.

네이버 Preferred(LangGraph). 지금까지의 단계(asr·extract·relations·evaluate)를 흩어진 함수가 아니라
**명시적 상태 그래프**로 엮어, 각 노드가 상태를 갱신하고 마지막 검증 노드가 **needs_human 게이트**를
건다(inspection-copilot supervisor 패턴 재사용). 임상 시스템의 요건 = "불확실하면 사람에게".

그래프:  asr → normalize → extract → relations → validate → END
검증 게이트(needs_human) 트리거:
  - 진단(diagnosis)을 못 뽑음(핵심 필드 결측)
  - 약물 recall 위험 신호 = 추출 약물이 0개
  - 관계가 비어 있음(구조 복원 실패)
게이트가 걸리면 자동 확정 대신 사람 검토로 라우팅(감사 근거 state에 기록).

**normalize 노드**(용어 정규화)는 ASR과 추출 사이에 놓인다 — 깨진 의료 용어를 사전 표준형으로
되돌려 뒷단이 볼 수 있게 만든다. 적용된 교정은 `corrections`로 상태에 남겨 **감사 추적**이 되게
한다(임상 시스템은 '무엇을 왜 바꿨는지'가 기록돼야 한다). normalizer=None이면 노드는 통과.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .asr import NoisyASR
from .extract import RuleExtractor
from .normalize import TermNormalizer
from .relations import extract_relations


class NoteState(TypedDict, total=False):
    clean: str            # 원본(정답) 전사 — 시뮬 입력
    terms: list           # 의료 용어(오류주입 편향용)
    transcript: str       # ASR 출력 전사
    normalized: str       # 용어 정규화를 거친 전사(추출 입력)
    corrections: list     # 적용된 교정 목록(감사 추적)
    note: Any             # ClinicalNote
    relations: set        # 관계 집합
    needs_human: bool
    reasons: list         # 게이트 사유(감사)


def build_graph(asr: NoisyASR | None = None, fuzzy: float = 0.7,
                normalizer: TermNormalizer | None = None):
    """상태 그래프 컴파일.

    asr=None이면 이미 transcript가 상태에 있다고 보고 ASR 노드는 통과.
    normalizer=None이면 정규화 노드는 전사를 그대로 흘려보낸다(기존 동작 불변).
    """
    extractor = RuleExtractor(fuzzy=fuzzy)

    def asr_node(state: NoteState) -> NoteState:
        if asr is not None and state.get("clean"):
            return {"transcript": asr.transcribe_text(state["clean"], state.get("terms", []))}
        return {"transcript": state.get("transcript", state.get("clean", ""))}

    def normalize_node(state: NoteState) -> NoteState:
        """ASR 후처리 — 깨진 의료 용어를 사전 표준형으로 복원(교정 이력은 감사용으로 보존)."""
        text = state["transcript"]
        if normalizer is None:
            return {"normalized": text, "corrections": []}
        fixed, corrs = normalizer.normalize(text)
        return {"normalized": fixed,
                "corrections": [{"span": c.span, "term": c.term, "score": c.score,
                                 "layer": c.layer} for c in corrs]}

    def extract_node(state: NoteState) -> NoteState:
        return {"note": extractor.extract(state["normalized"])}

    def relations_node(state: NoteState) -> NoteState:
        return {"relations": extract_relations(state["note"], state["normalized"])}

    def validate_node(state: NoteState) -> NoteState:
        note, rels = state["note"], state["relations"]
        reasons = []
        if not note.diagnosis:
            reasons.append("진단 필드 결측")
        if not note.medications:
            reasons.append("추출 약물 0개(약물 누락 위험)")
        if not rels:
            reasons.append("관계 복원 실패")
        return {"needs_human": bool(reasons), "reasons": reasons}

    g = StateGraph(NoteState)
    g.add_node("asr", asr_node)
    g.add_node("normalize", normalize_node)
    g.add_node("extract", extract_node)
    g.add_node("relations", relations_node)
    g.add_node("validate", validate_node)
    g.set_entry_point("asr")
    g.add_edge("asr", "normalize")
    g.add_edge("normalize", "extract")
    g.add_edge("extract", "relations")
    g.add_edge("relations", "validate")
    g.add_edge("validate", END)
    return g.compile()


def run_encounter(clean: str, terms: list, asr: NoisyASR | None = None,
                  fuzzy: float = 0.7, normalizer: TermNormalizer | None = None) -> NoteState:
    """단일 진료 건을 그래프로 처리해 최종 상태 반환."""
    app = build_graph(asr=asr, fuzzy=fuzzy, normalizer=normalizer)
    return app.invoke({"clean": clean, "terms": terms})
