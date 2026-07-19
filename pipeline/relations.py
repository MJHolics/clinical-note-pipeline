"""관계 추출(RE) + 부정(negation) + 시점(temporal) — NER을 넘어선 임상 정보추출.

네이버 Required가 "Entity Recognition **/ Relation Extraction** 등의 NLP"라 개체만이 아니라 **관계**를
복원해야 한다. 진료 대화에서:
  - INDICATES : 증상 → 진단(증상이 진단을 시사)
  - TREATS    : 약물 → 진단(약물이 진단을 치료)
  - ONSET     : 진단 → 발병 시점(temporal IE)
그리고 **부정(negation)** — '통증은 없습니다' 같은 pertinent negative를 그냥 개체로 뽑으면 오진이므로,
부정된 증상은 기록·관계에서 제외해야 한다(임상 안전 직결).

순수 함수만(전사·개체 목록 입력 → 집합 출력). extract.py를 top-level import하지 않아 순환참조 없음.
"""
from __future__ import annotations

import re

NEGATION_CUES = ("없", "않", "아니", "아닙", "부정", "느껴지지")

# 발병 시점(temporal) 패턴 — 상대 시점 표현
_ONSET_PATTERNS = [
    r"\d+\s*일\s*전", r"\d+\s*주\s*전", r"\d+\s*개월\s*전", r"\d+\s*달\s*전",
    r"\d+\s*년\s*전", r"\d+\s*시간\s*전",
    r"한\s*달\s*전", r"일주일\s*전", r"이틀\s*전", r"어제", r"그저께", r"오늘", r"봄철",
]
_ONSET_RE = re.compile("|".join(_ONSET_PATTERNS))


def detect_negations(text: str, symptoms: list[str], window: int = 16) -> set[str]:
    """전사에서 부정된 증상 집합. 증상 언급 뒤 window 문자 내 부정 단서가 있으면 부정으로 판정.

    한국어 부정은 후치(증상은 '없습니다')라 증상 위치 이후를 본다. 어느 한 출현이라도 부정
    단서를 동반하면 부정.
    """
    negated: set[str] = set()
    for sym in symptoms:
        start = 0
        while True:
            i = text.find(sym, start)
            if i < 0:
                break
            seg = text[i + len(sym): i + len(sym) + window]
            if any(cue in seg for cue in NEGATION_CUES):
                negated.add(sym)
                break
            start = i + len(sym)
    return negated


def extract_onset(text: str) -> str:
    """전사에서 발병 시점 표현을 추출(첫 매칭). 없으면 빈 문자열. 공백 정규화."""
    m = _ONSET_RE.search(text)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group()).strip()


def extract_relations(note, transcript: str) -> set[tuple[str, str, str]]:
    """추출된 note(개체) + 전사 → 관계 집합.

    note는 duck-typed(.symptoms/.medications/.diagnosis) — 순환참조 회피. 부정 증상은 이미 note에서
    제외됐다고 가정(negation-aware 추출기). ONSET은 전사에서 시점 패턴으로 복원.
    """
    dx = note.diagnosis
    rels: set[tuple[str, str, str]] = set()
    if dx:
        for s in note.symptoms:
            rels.add((s, "INDICATES", dx))
        for m in note.medications:
            rels.add((m, "TREATS", dx))
        onset = extract_onset(transcript)
        if onset:
            rels.add((dx, "ONSET", onset))
    return rels


def relation_prf(gold: set, pred: set) -> dict:
    """관계 집합 P/R/F1(순수) — 튜플 동일성 기준."""
    tp = len(gold & pred)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "n_gold": len(gold), "n_pred": len(pred)}


def negation_recall(gold_negated: set[str], detected_negated: set[str]) -> float:
    """부정 소견을 올바로 부정으로 잡은 비율(놓치면 오진 위험)."""
    if not gold_negated:
        return 1.0
    return len(gold_negated & detected_negated) / len(gold_negated)
