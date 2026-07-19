"""평가 단계 — 추출된 진료기록 vs 정답(gold)의 개체 P/R/F1 + 환각율.

의료 문서 자동화의 두 실패 모드를 분리 측정:
  - **누락(recall↓)**: 실제 있던 증상·약물을 놓침 — 임상적으로 더 위험(처방 누락).
  - **환각(precision↓ / hallucination)**: 대화에 없던 개체를 지어냄 — 오진 유발.
전사에도 gold에도 없는 개체를 pred가 냈다면 순수 환각으로 따로 센다(LLM 추출기 평가에 중요).
"""
from __future__ import annotations

from .dialogue import Case
from .extract import ClinicalNote


def prf(gold: set[str], pred: set[str]) -> dict:
    """집합 기반 precision/recall/F1(순수)."""
    tp = len(gold & pred)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "n_gold": len(gold), "n_pred": len(pred)}


def gold_note(case: Case) -> ClinicalNote:
    return ClinicalNote(symptoms=case.symptoms, medications=case.medications,
                        diagnosis=case.diagnosis)


def score_note(case: Case, pred: ClinicalNote, transcript: str = "") -> dict:
    """케이스 단위 채점 — 전체 개체 F1 + 필드별 recall + 환각율."""
    gold = gold_note(case)
    overall = prf(gold.entities(), pred.entities())
    sx = prf(set(case.symptoms), set(pred.symptoms))
    meds = prf(set(case.medications), set(pred.medications))
    # 환각: pred 개체 중 gold에도 전사에도 없는 것(순수 지어냄)
    hall = 0
    if pred.entities():
        for e in pred.entities():
            if e not in gold.entities() and (transcript and e not in transcript):
                hall += 1
    hall_rate = hall / len(pred.entities()) if pred.entities() else 0.0
    return {
        "case_id": case.case_id,
        "f1": overall["f1"], "precision": overall["precision"], "recall": overall["recall"],
        "symptom_recall": sx["recall"], "med_recall": meds["recall"],
        "hallucination_rate": round(hall_rate, 4),
    }


def aggregate(scores: list[dict]) -> dict:
    """케이스 점수들을 평균."""
    if not scores:
        return {}
    keys = ["f1", "precision", "recall", "symptom_recall", "med_recall", "hallucination_rate"]
    return {k: round(sum(s[k] for s in scores) / len(scores), 4) for k in keys}
