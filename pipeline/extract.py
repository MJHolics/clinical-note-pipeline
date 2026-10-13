"""추출 단계 — 전사(transcript) → 구조화 진료기록(SOAP 개체).

진료기록 생성의 코어: 대화/전사에서 **개체(증상·약물·진단)를 인식·추출**해 구조화한다(NER/RE 성격).
두 백엔드:
  - RuleExtractor  : 의료 렉시콘 기반 결정적 추출(+선택적 퍼지 매칭으로 경미한 ASR 오류 복원).
                     키·네트워크 없이 하네스에서 재현 — 오류 전파 실험의 통제 추출기.
  - LLMExtractor   : 주입형 LLM(무료 Gemini)로 추출. 실제 운영형 경로(스텁 주입해 테스트 가능).

퍼지 매칭은 **정직한 절충**을 실증한다 — 글자 하나 틀린 약물명은 복원하나, 통째로 다른 단어로
인식된 용어는 못 살린다(뒤 evaluate에서 수치로 드러남).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dialogue import CASES
from .relations import detect_negations


@dataclass(frozen=True)
class ClinicalNote:
    symptoms: tuple[str, ...] = ()
    medications: tuple[str, ...] = ()
    diagnosis: str = ""

    def entities(self) -> set[str]:
        s = set(self.symptoms) | set(self.medications)
        if self.diagnosis:
            s.add(self.diagnosis)
        return s


def build_lexicon() -> dict[str, list[str]]:
    """전체 케이스에서 개체 렉시콘 구축(증상/약물/진단). 실무의 사전·온톨로지 자리."""
    lex = {"symptoms": set(), "medications": set(), "diagnosis": set()}
    for c in CASES:
        lex["symptoms"].update(c.symptoms)
        lex["medications"].update(c.medications)
        lex["diagnosis"].add(c.diagnosis)
    return {k: sorted(v) for k, v in lex.items()}


def _char_ratio(a: str, b: str) -> float:
    """두 문자열의 문자 다중집합 유사도(0~1) — 퍼지 매칭용, 순수."""
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    return inter / max(len(a), len(b))


def match_entity(term: str, text: str, fuzzy: float = 0.0) -> bool:
    """term이 text에 존재하는가. fuzzy>0이면 어절 단위로 문자 유사도 임계 매칭(ASR 오류 복원)."""
    if term in text:
        return True
    if fuzzy <= 0:
        return False
    words = text.split()
    tlen = len(term.split())
    for i in range(len(words) - tlen + 1):        # 다어절 용어는 인접 어절 결합해 비교
        cand = "".join(words[i:i + tlen])
        if _char_ratio(term.replace(" ", ""), cand) >= fuzzy:
            return True
    return False


def _drop_nested(matched: list[str]) -> tuple[str, ...]:
    """중첩 개체 해소 — longest-match-wins. 다른 더 긴 매칭의 부분문자열이면 제거.

    예: '야간 기침'이 매칭되면 그 부분인 '기침'은 버린다(nested entity, NER 표준 처리).
    이 규칙이 없으면 '기침'이 오탐(false positive)으로 precision을 떨어뜨린다.
    """
    keep = []
    for t in matched:
        nested = any(t != o and t.replace(" ", "") in o.replace(" ", "") for o in matched)
        if not nested:
            keep.append(t)
    return tuple(keep)


class RuleExtractor:
    """렉시콘 기반 결정적 추출기. fuzzy=0이면 정확 매칭, >0이면 퍼지 복원 활성.

    중첩 개체는 longest-match-wins로 해소('야간 기침' 있으면 '기침' 제거).
    """
    def __init__(self, fuzzy: float = 0.0):
        self.lex = build_lexicon()
        self.fuzzy = fuzzy

    def extract(self, transcript: str) -> ClinicalNote:
        found = [t for t in self.lex["symptoms"] if match_entity(t, transcript, self.fuzzy)]
        # 부정(negation) 처리: '통증은 없습니다'로 부정된 증상은 기록에서 제외(임상 안전).
        negated = detect_negations(transcript, found)
        sx = _drop_nested([t for t in found if t not in negated])
        meds = _drop_nested([t for t in self.lex["medications"] if match_entity(t, transcript, self.fuzzy)])
        dx = next((t for t in self.lex["diagnosis"] if match_entity(t, transcript, self.fuzzy)), "")
        return ClinicalNote(symptoms=sx, medications=meds, diagnosis=dx)


class LLMExtractor:
    """주입형 LLM 추출기 — client는 .complete(prompt)->str 만 있으면 됨(Gemini/스텁 무관)."""
    def __init__(self, client):
        self.client = client
        self.lex = build_lexicon()

    def _prompt(self, transcript: str) -> str:
        return (
            "다음 진료 대화 전사에서 증상·약물·진단을 JSON으로 추출하세요.\n"
            '형식: {"symptoms": [], "medications": [], "diagnosis": ""}\n\n'
            f"전사: {transcript}\n"
        )

    def extract(self, transcript: str) -> ClinicalNote:
        import json
        raw = self.client.complete(self._prompt(transcript))
        try:
            d = json.loads(raw)
        except Exception:
            return ClinicalNote()
        return ClinicalNote(
            symptoms=tuple(d.get("symptoms", [])),
            medications=tuple(d.get("medications", [])),
            diagnosis=d.get("diagnosis", "") or "",
        )
