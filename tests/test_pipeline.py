"""파이프라인 순수함수·단계·오류전파 테스트."""
from pipeline.asr import NoisyASR, term_error_rate, wer
from pipeline.dialogue import CASES, transcript_of
from pipeline.evaluate import gold_note, prf, score_note
from pipeline.extract import ClinicalNote, RuleExtractor, match_entity
from pipeline.relations import (
    detect_negations,
    extract_onset,
    extract_relations,
    relation_prf,
)


def test_wer_basics():
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b c d", "a b x d") == 0.25          # 1/4 치환
    assert wer("", "") == 0.0
    assert wer("a", "") == 1.0


def test_term_error_rate():
    terms = ["metformin", "기침"]
    assert term_error_rate(terms, "환자가 기침 하고 metformin 복용") == 0.0
    assert term_error_rate(terms, "환자가 기침 함") == 0.5    # metformin 누락
    assert term_error_rate([], "아무거나") == 0.0


def test_prf_perfect_and_miss():
    p = prf({"a", "b"}, {"a", "b"})
    assert p["f1"] == 1.0
    m = prf({"a", "b"}, {"a"})
    assert m["recall"] == 0.5 and m["precision"] == 1.0


def test_clean_extraction_is_perfect():
    """오류 없는 전사는 정답 개체를 완벽 복원(파이프라인 상한 확인)."""
    ext = RuleExtractor(fuzzy=0.0)
    for case in CASES:
        pred = ext.extract(transcript_of(case))
        sc = score_note(case, pred, transcript=transcript_of(case))
        assert sc["f1"] == 1.0, f"{case.case_id} clean F1={sc['f1']}"


def test_asr_noise_degrades_then_propagates():
    """ASR 오류 주입 → 전사 오류 발생 → 추출 F1 하락(전파)."""
    case = CASES[0]
    clean = transcript_of(case)
    noisy = NoisyASR(general_er=0.05, term_er=0.6).transcribe_text(clean, case.medical_terms())
    assert noisy != clean                              # 실제로 오염됨
    assert term_error_rate(case.medical_terms(), noisy) > 0.0
    pred = RuleExtractor(fuzzy=0.0).extract(noisy)
    sc = score_note(case, pred, transcript=noisy)
    assert sc["f1"] < 1.0                               # 품질 전파 하락


def test_deterministic():
    """같은 설정은 같은 결과(재현성)."""
    case = CASES[2]
    a = NoisyASR(term_er=0.3).transcribe_text(transcript_of(case), case.medical_terms())
    b = NoisyASR(term_er=0.3).transcribe_text(transcript_of(case), case.medical_terms())
    assert a == b


def test_fuzzy_recovers_some_errors():
    """퍼지 매칭이 경미한 글자 오류를 정확 매칭보다 더 복원."""
    # 'metformin'에서 한 글자 빠진 전사
    text = "환자 metformi 복용 기침 콧물"
    assert not match_entity("metformin", text, fuzzy=0.0)     # 정확은 실패
    assert match_entity("metformin", text, fuzzy=0.7)         # 퍼지는 복원


def test_hallucination_counted():
    """전사·gold에 없는 개체를 pred가 내면 환각으로 집계."""
    case = CASES[0]
    fake = ClinicalNote(symptoms=("기침",), medications=("존재하지않는약",), diagnosis="")
    sc = score_note(case, fake, transcript="기침 콧물")
    assert sc["hallucination_rate"] > 0.0


# ── 관계 추출(RE) · 부정 · 시점 ──────────────────────────────────────
def test_negation_excludes_symptom():
    """부정된 증상('호흡곤란은 없습니다')은 추출에서 제외(임상 안전)."""
    case = CASES[0]                                  # negated_symptoms=('호흡곤란',)
    t = transcript_of(case)
    assert "호흡곤란" in t                             # 문자열은 존재
    note = RuleExtractor().extract(t)
    assert "호흡곤란" not in note.symptoms             # 그러나 부정 처리로 제외
    assert detect_negations(t, ["호흡곤란"]) == {"호흡곤란"}


def test_negation_keeps_positive():
    """긍정 증상은 부정으로 오분류하지 않음."""
    case = CASES[0]
    t = transcript_of(case)
    assert detect_negations(t, ["기침"]) == set()     # '기침'은 긍정


def test_onset_extraction():
    assert extract_onset("3일 전부터 기침") == "3일 전"
    assert extract_onset("어제부터 아팠어요") == "어제"
    assert extract_onset("증상 없음") == ""


def test_relations_clean_perfect():
    """오류 없는 전사는 정답 관계를 완벽 복원(RE 상한)."""
    for case in CASES:
        t = transcript_of(case)
        note = RuleExtractor().extract(t)
        pred = extract_relations(note, t)
        rel = relation_prf(case.gold_relations(), pred)
        assert rel["f1"] == 1.0, f"{case.case_id} rel F1={rel['f1']}"


def test_relation_prf_partial():
    gold = {("a", "R", "x"), ("b", "R", "x")}
    pred = {("a", "R", "x")}
    r = relation_prf(gold, pred)
    assert r["recall"] == 0.5 and r["precision"] == 1.0


def test_score_includes_relation_and_negation():
    case = CASES[0]
    t = transcript_of(case)
    sc = score_note(case, RuleExtractor().extract(t), t)
    assert sc["relation_f1"] == 1.0
    assert sc["negation_recall"] == 1.0


# ── LangGraph 오케스트레이션 ─────────────────────────────────────────
def test_orchestrate_clean_passes_gate():
    import pytest
    pytest.importorskip("langgraph")
    from pipeline.asr import NoisyASR
    from pipeline.orchestrate import run_encounter
    case = CASES[0]
    s = run_encounter(transcript_of(case), case.medical_terms(),
                      asr=NoisyASR(term_er=0.0, general_er=0.0))
    assert s["needs_human"] is False           # 깨끗하면 자동 확정
    assert s["note"].diagnosis and s["relations"]


def test_orchestrate_severe_noise_needs_human():
    import pytest
    pytest.importorskip("langgraph")
    from pipeline.asr import NoisyASR
    from pipeline.orchestrate import run_encounter
    case = CASES[0]
    s = run_encounter(transcript_of(case), case.medical_terms(),
                      asr=NoisyASR(term_er=0.95, general_er=0.3), fuzzy=0.0)
    assert s["needs_human"] is True            # 심한 오류 → 사람 검토 게이트
    assert s["reasons"]
