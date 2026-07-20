"""용어 정규화 테스트 — 자모 분해·음차 골격·복원 능력·**안전 불변식**.

이 층의 위험은 성능이 아니라 **안전**이다. 정상 텍스트를 망가뜨리거나 없던 개체를 만들어내면
임상에서 오진·과잉처방으로 직결된다. 그래서 복원 능력만큼(아니 그보다 더) 다음을 강제한다:
  - 정상 전사는 추출 결과가 불변(F1 1.0 유지)
  - 사전에 없는 문자열은 절대 생성하지 않음(환각 구조적 불가)
  - 정상 전사에서 유해 오교정 0건(운영점 회귀 방지)
"""
import re

from pipeline.dialogue import CASES, transcript_of
from pipeline.evaluate import score_note
from pipeline.extract import RuleExtractor, build_lexicon
from pipeline.normalize import (
    TermNormalizer,
    decompose,
    jamo_similarity,
    skeleton_hangul,
    skeleton_latin,
    skeleton_similarity,
    to_jamo,
)


def _all_terms():
    lex = build_lexicon()
    return [*lex["symptoms"], *lex["medications"], *lex["diagnosis"]]


def _norm_text(s):
    s = re.sub(r"[^\w가-힣 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# ── 자모 분해 (L1 기반) ──────────────────────────────────────────────
def test_decompose_syllable():
    assert decompose("각") == ("ㄱ", "ㅏ", "ㄱ")
    assert decompose("가") == ("ㄱ", "ㅏ", "")      # 종성 없음
    assert decompose("A") is None                   # 비한글


def test_to_jamo_keeps_nonhangul():
    assert to_jamo("가") == "ㄱㅏ"
    assert to_jamo("a1") == "a1"                    # 영문·숫자는 보존
    assert to_jamo("가 나") == "ㄱㅏㄴㅏ"             # 공백 제거


def test_jamo_beats_char_similarity_on_real_asr_error():
    """이 층의 존재 이유 — 실제 Whisper 오류 '배뇨통→배뉴통'(중성 ㅛ→ㅠ 하나).

    문자 다중집합 유사도는 0.667(기존 퍼지 임계 0.7 미달 → 놓침)인데,
    자모 단위로는 1/7 치환이라 0.857로 잡힌다.
    """
    from collections import Counter

    gold, asr = "배뇨통", "배뉴통"
    ca, cb = Counter(gold), Counter(asr)
    char_sim = sum((ca & cb).values()) / max(len(gold), len(asr))
    assert char_sim < 0.7                            # 기존 방식은 놓친다
    assert jamo_similarity(gold, asr) > 0.85         # 자모는 잡는다


def test_jamo_similarity_bounds():
    assert jamo_similarity("기침", "기침") == 1.0
    assert jamo_similarity("", "기침") == 0.0


# ── 음차 골격 (L2 기반) ──────────────────────────────────────────────
def test_skeleton_drops_vowels_and_folds_phonemes():
    assert skeleton_latin("salbutamol") == "slptml"  # 모음 제거 + b→p, r→l
    assert skeleton_latin("cetirizine") == "ktljn"   # c→k, r→l, z→j
    assert skeleton_hangul("셀뷰터말") == "slptml"


def test_skeleton_bridges_script_mismatch():
    """영문 성분명↔한글 음차는 자모로는 0.0(스크립트 불일치) — 골격이 유일한 다리."""
    assert jamo_similarity("salbutamol", "셀뷰터말") == 0.0
    assert skeleton_similarity("salbutamol", "셀뷰터말") == 1.0
    assert skeleton_similarity("amlodipine", "앰로디핀") == 1.0


def test_skeleton_rejects_common_korean_words():
    """일반 어절이 성분명으로 끌려가면 안 된다(없던 약 생성 = 임상 위험)."""
    nz = TermNormalizer(_all_terms())
    for word in ["왔습니다", "없습니다", "보입니다", "감사합니다", "처방해"]:
        out, corrs = nz.normalize(word)
        assert out == word and not corrs, f"'{word}'가 교정됨: {corrs}"


# ── 복원 능력 ────────────────────────────────────────────────────────
def test_restores_real_whisper_errors():
    """실제 Whisper가 낸 오인식을 사전 표준형으로 되돌리는가(운영점 기준)."""
    nz = TermNormalizer(_all_terms())
    for bad, want in [("호흡골란", "호흡곤란"),        # 자모 0.909
                      ("속스림", "속쓰림"),            # 자모 0.875
                      ("앰로디핀", "amlodipine"),      # 골격 1.000
                      ("매트 포어 민", "metformin")]:  # 골격 0.833
        out, corrs = nz.normalize(bad)
        assert want in out, f"'{bad}' → '{out}' (기대: {want}), 교정={corrs}"


def test_safety_threshold_costs_some_recall():
    """안전 제약의 **대가**를 명시적으로 고정한다(정직한 트레이드오프).

    '콩물→콧물'은 자모 유사도 0.833이라 운영점(0.85)에서는 복원되지 않는다. 임계를 0.80으로
    낮추면 복원되지만, 그 설정은 정상 전사에서 유해 오교정을 만든다(스윕상 2건).
    즉 **복원 못 한 콧물 하나는 '없는 약을 만들지 않기 위해 치른 비용'**이다.
    """
    assert 0.80 < jamo_similarity("콧물", "콩물") < 0.85
    safe = TermNormalizer(_all_terms())                          # 0.85 — 운영점
    assert "콧물" not in safe.normalize("콩물")[0]                # 포기됨
    loose = TermNormalizer(_all_terms(), jamo_threshold=0.80)    # 더 공격적
    assert "콧물" in loose.normalize("콩물")[0]                   # 복원되지만 안전은 잃는다


def test_layers_are_orthogonal():
    """L1은 한국어 용어를, L2는 영문 성분명을 담당 — 서로 못 덮는다(직교)."""
    terms = _all_terms()
    jamo_only = TermNormalizer(terms, layers=("jamo",))
    skel_only = TermNormalizer(terms, layers=("skeleton",))
    # 한국어 용어: L1만 복원(골격층은 한글 용어를 대상으로 삼지 않는다)
    assert "속쓰림" in jamo_only.normalize("속스림")[0]
    assert "속쓰림" not in skel_only.normalize("속스림")[0]
    # 영문 성분명 음차: L2만 복원(자모로는 스크립트가 달라 0.0)
    assert "amlodipine" in skel_only.normalize("앰로디핀")[0]
    assert "amlodipine" not in jamo_only.normalize("앰로디핀")[0]


# ── 안전 불변식 ──────────────────────────────────────────────────────
def test_clean_transcript_extraction_unchanged():
    """정상 전사는 정규화를 거쳐도 추출 결과가 완벽(F1 1.0) — 파이프라인 상한 불변."""
    nz = TermNormalizer(_all_terms())
    ext = RuleExtractor(fuzzy=0.0)
    for case in CASES:
        clean = _norm_text(transcript_of(case))
        normalized, _ = nz.normalize(clean)
        sc = score_note(case, ext.extract(normalized), transcript=normalized)
        assert sc["f1"] == 1.0, f"{case.case_id}: 정규화가 정상 전사를 훼손 (F1={sc['f1']})"


def test_never_invents_terms_outside_lexicon():
    """정규화는 **사전 표준형으로만** 되돌린다 — 환각이 구조적으로 불가능."""
    terms = set(_all_terms())
    nz = TermNormalizer(_all_terms())
    for case in CASES:
        _, corrs = nz.normalize(_norm_text(transcript_of(case)))
        for c in corrs:
            assert c.term in terms, f"사전에 없는 용어 생성: {c.term}"


def test_no_harmful_overcorrection_at_operating_point():
    """운영점(자모 0.85/골격 0.70)에서 정상 전사의 유해 오교정 0건 — 회귀 방지.

    유해 = 그 케이스에 없는 개체를 만들어내는 교정. 임계를 낮추면 이 값이 급증하므로
    (0.70에서 7건, 0.60에서 34건) 기본값이 안전 쪽에 있음을 테스트가 고정한다.
    """
    nz = TermNormalizer(_all_terms())
    harmful = []
    for case in CASES:
        gold = set(case.medical_terms())
        _, corrs = nz.normalize(_norm_text(transcript_of(case)))
        for c in corrs:
            benign = c.term in gold and c.term.replace(" ", "") in c.span.replace(" ", "")
            if not benign:
                harmful.append((case.case_id, c.span, c.term))
    assert not harmful, f"유해 오교정 발생: {harmful}"


def test_graph_records_corrections_for_audit():
    """그래프에 정규화 노드가 실제로 끼어들고, 교정 이력이 감사용으로 상태에 남는가.

    임상 시스템은 자동 교정을 '조용히' 하면 안 된다 — 무엇을 무엇으로 왜 바꿨는지가
    기록돼야 검토·책임 추적이 가능하다.
    """
    import pytest
    pytest.importorskip("langgraph")
    from pipeline.asr import NoisyASR
    from pipeline.orchestrate import run_encounter

    case = CASES[0]
    nz = TermNormalizer(_all_terms())
    s = run_encounter(transcript_of(case), case.medical_terms(),
                      asr=NoisyASR(term_er=0.0, general_er=0.0), normalizer=nz)
    assert "normalized" in s and "corrections" in s
    for c in s["corrections"]:                       # 감사 레코드 형태 강제
        assert {"span", "term", "score", "layer"} <= set(c)
    assert s["needs_human"] is False                 # 정상 건은 여전히 자동 확정


def test_normalization_preserves_negation():
    """정규화가 부정 소견을 되살려도 부정 처리가 여전히 걸러야 한다.

    '호흡골란 증상은 없습니다' → 정규화로 '호흡곤란'이 복원되지만, 부정 단서가 뒤에 있으므로
    기록에서는 제외돼야 한다(정규화가 오진을 만들지 않음).
    """
    nz = TermNormalizer(_all_terms())
    text, _ = nz.normalize("그런데 호흡골란 증상은 없습니다")
    assert "호흡곤란" in text                          # 복원은 됐고
    note = RuleExtractor(fuzzy=0.0).extract(text)
    assert "호흡곤란" not in note.symptoms             # 기록에는 안 들어간다
