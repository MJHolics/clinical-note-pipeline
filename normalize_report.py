"""용어 정규화 실측 — "재학습보다 용어 정규화가 첫 수"라는 결론을 실제로 검증한다.

이 프로젝트는 ④에서 ASR 오류가 진료기록으로 전파되는 걸 측정하고 "ASR 후처리(용어 정규화)가
재학습보다 비용 대비 효과적인 첫 수"라고 *주장만* 했다. 이 스크립트가 그 주장을 닫는다:

  1) **실제 Whisper 전사**(outputs/real_asr.json 캐시)에 정규화를 적용해 개체·관계 F1 회복 측정
     — 시뮬이 아니라 실모델이 실제로 낸 오류에서 재현.
  2) **층별 ablation**(L1 자모 / L2 음차 골격 / L1+L2 / 기존 퍼지)로 무엇이 무엇을 고치는지 분리.
  3) **오교정(false correction) 측정** — 정상(clean) 전사를 정규화했을 때 텍스트가 변하는지.
     복원만 보고하고 부작용을 안 보면 정직하지 않다.
  4) **임계값 트레이드오프 스윕** — 임계를 낮추면 복원↑ 오교정↑. 어디서 끊어야 하는가.

오디오·키·네트워크 불필요(캐시된 전사 + 결정적 정규화). 재현: python normalize_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pipeline.dialogue import CASES, transcript_of
from pipeline.evaluate import aggregate, score_note
from pipeline.extract import RuleExtractor, build_lexicon
from pipeline.normalize import TermNormalizer

OUT = Path(__file__).resolve().parent / "outputs"
REAL = OUT / "real_asr.json"


def all_terms() -> list[str]:
    """전체 의료 렉시콘(증상+약물+진단) — 실무의 약물사전·온톨로지 자리.

    주의: 케이스별이 아니라 *전체* 사전이라 다른 케이스의 용어로 잘못 끌려갈 위험이 있다.
    오교정 측정이 필요한 이유(사전이 클수록 오교정 표면적도 커진다).
    """
    lex = build_lexicon()
    return [*lex["symptoms"], *lex["medications"], *lex["diagnosis"]]


def load_real() -> list[dict] | None:
    if not REAL.exists():
        return None
    return json.loads(REAL.read_text(encoding="utf-8"))["rows"]


def _norm_text(s: str) -> str:
    """real_asr.py와 동일한 비교용 정규화(문장부호 제거·소문자·공백 단일화)."""
    import re
    s = re.sub(r"[^\w가-힣 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def eval_config(rows: list[dict], normalizer: TermNormalizer | None, fuzzy: float) -> dict:
    """한 설정(정규화기 유무 × 퍼지)으로 실제 Whisper 전사 전건을 채점."""
    by_id = {c.case_id: c for c in CASES}
    ext = RuleExtractor(fuzzy=fuzzy)
    scores, ters, n_corr = [], [], 0
    for r in rows:
        case = by_id.get(r["case_id"])
        if case is None:
            continue
        hyp = _norm_text(r["hyp"])
        if normalizer is not None:
            hyp, corrs = normalizer.normalize(hyp)
            n_corr += len(corrs)
        # 정규화 후 용어 보존율(=term error rate 재측정)
        terms = case.medical_terms()
        ters.append(sum(1 for t in terms if t not in hyp) / len(terms))
        scores.append(score_note(case, ext.extract(hyp), transcript=hyp))
    agg = aggregate(scores)
    agg["term_er"] = round(sum(ters) / len(ters), 4)
    agg["n_corrections"] = n_corr
    return agg


def real_ablation(rows: list[dict]) -> dict:
    """실제 Whisper 전사에서 층별 ablation."""
    terms = all_terms()
    configs = [
        ("① 정규화 없음 (exact)", None, 0.0),
        ("② 기존 퍼지 매칭(문자)", None, 0.7),
        ("③ L1 자모 정규화만", TermNormalizer(terms, layers=("jamo",)), 0.0),
        ("④ L2 음차 골격만", TermNormalizer(terms, layers=("skeleton",)), 0.0),
        ("⑤ L1+L2 (제안)", TermNormalizer(terms), 0.0),
        ("⑥ L1+L2 + 퍼지", TermNormalizer(terms), 0.7),
    ]
    print("=" * 92)
    print("실제 Whisper 전사(TTS→faster-whisper base, 8케이스)에 용어 정규화 적용 — 층별 ablation")
    print("=" * 92)
    print(f"{'설정':<24} | {'용어오류율':>9} {'개체 F1':>8} {'약물rec':>8} "
          f"{'관계 F1':>8} {'환각':>6} {'교정수':>6}")
    print("-" * 92)
    results = {}
    for name, nz, fz in configs:
        a = eval_config(rows, nz, fz)
        results[name] = a
        print(f"{name:<24} | {a['term_er']:>9.3f} {a['f1']:>8.3f} {a['med_recall']:>8.3f} "
              f"{a['relation_f1']:>8.3f} {a['hallucination_rate']:>6.3f} {a['n_corrections']:>6d}")
    print("-" * 92)
    base, prop = results["① 정규화 없음 (exact)"], results["⑤ L1+L2 (제안)"]
    fuzzy = results["② 기존 퍼지 매칭(문자)"]
    print(f"[요약] 개체 F1 {base['f1']:.3f} → 퍼지 {fuzzy['f1']:.3f} → 정규화 {prop['f1']:.3f}  "
          f"(정규화가 퍼지 대비 +{prop['f1'] - fuzzy['f1']:.3f})")
    print(f"       관계 F1 {base['relation_f1']:.3f} → {prop['relation_f1']:.3f}  ·  "
          f"용어오류율 {base['term_er']:.3f} → {prop['term_er']:.3f}")
    return results


def count_overcorrections(nz: TermNormalizer) -> tuple[int, int, list]:
    """정상(clean) 전사에서 유해/무해 교정을 분리해 센다.

    clean 전사는 이미 모든 용어가 표준형이므로 여기서 일어나는 교정은 전부 '불필요'하다. 다만
    그 성격이 다르다 — 이걸 뭉뚱그리면 지표가 거짓말을 한다:

      **무해**: 목표 용어가 그 케이스의 정답 용어이고 구간이 그 용어를 이미 품고 있는 경우.
               예 '급성 상기도 감염으로' → '급성 상기도 감염' (조사 제거·띄어쓰기 수렴).
               개체는 그대로라 추출 결과가 바뀌지 않는다.
      **유해**: 목표 용어가 그 케이스에 없는 개체인 경우. 예 '매일 혈압' → 'amlodipine'.
               **없던 개체를 기록에 만들어내므로 임상적으로 위험**(오진·과잉처방) — 이것만이
               진짜 오교정이고, 운영점은 이 값으로 정해야 한다.
    """
    harmful = harmless = 0
    examples = []
    for case in CASES:
        gold = set(case.medical_terms())
        clean = _norm_text(transcript_of(case))
        _, corrs = nz.normalize(clean)
        for c in corrs:
            benign = c.term in gold and c.term.replace(" ", "") in c.span.replace(" ", "")
            if benign:
                harmless += 1
            else:
                harmful += 1
                examples.append((case.case_id, c.span, c.term, c.score, c.layer))
    return harmful, harmless, examples


def overcorrection() -> dict:
    """오교정 측정 — 정상 전사가 오염되는가(복원만 보고하고 부작용을 숨기면 정직하지 않다)."""
    nz = TermNormalizer(all_terms())
    print("\n" + "=" * 92)
    print("오교정(false correction) 점검 — 정상 전사는 그대로 보존되는가")
    print("=" * 92)
    harmful, harmless, examples = count_overcorrections(nz)
    total_tokens = sum(len(_norm_text(transcript_of(c)).split()) for c in CASES)
    print(f"정상 전사 {len(CASES)}건 · 총 {total_tokens} 어절")
    print(f"  · 유해 오교정(없던 개체 생성): **{harmful}**   ← 임상 위험, 운영점 결정 기준")
    print(f"  · 무해 교정(조사 제거 등 정답 용어로 수렴): {harmless}")
    for cid, span, term, score, layer in examples[:10]:
        print(f"   ⚠ [{cid}] '{span}' → '{term}' (유사도 {score:.3f}, {layer})")
    if not examples:
        print("   ✅ 유해 오교정 0건 — 정규화가 없던 개체를 만들어내지 않음.")
    return {"harmful": harmful, "harmless": harmless, "tokens": total_tokens,
            "examples": [{"case": c, "span": s, "term": t, "score": sc, "layer": l}
                         for c, s, t, sc, l in examples]}


def threshold_sweep(rows: list[dict]) -> dict:
    """임계값 트레이드오프 — 낮출수록 복원↑ 오교정↑. 어디서 끊는가."""
    print("\n" + "=" * 92)
    print("임계값 스윕 — 복원(개체 F1)과 오교정의 트레이드오프")
    print("=" * 92)
    print(f"{'자모임계':>8} {'골격임계':>8} | {'개체 F1':>8} {'약물rec':>8} {'관계 F1':>8} "
          f"| {'유해오교정':>10}")
    print("-" * 92)
    terms = all_terms()
    out = []
    for jt, st in [(0.60, 0.45), (0.65, 0.50), (0.70, 0.55), (0.75, 0.60),
                   (0.80, 0.65), (0.85, 0.70), (0.90, 0.80)]:
        nz = TermNormalizer(terms, jamo_threshold=jt, skeleton_threshold=st)
        a = eval_config(rows, nz, 0.0)
        harmful, _, _ = count_overcorrections(nz)
        print(f"{jt:>8.2f} {st:>8.2f} | {a['f1']:>8.3f} {a['med_recall']:>8.3f} "
              f"{a['relation_f1']:>8.3f} | {harmful:>10d}")
        out.append({"jamo": jt, "skeleton": st, "f1": a["f1"], "med_recall": a["med_recall"],
                    "relation_f1": a["relation_f1"], "harmful_overcorrections": harmful})
    print("-" * 92)
    # 운영점은 하드코딩이 아니라 **제약 하 최적화**로 고른다:
    #   제약 = 유해 오교정 0건(없던 개체를 만들지 않을 것, 임상 안전) → 그 안에서 개체 F1 최대.
    safe = [r for r in out if r["harmful_overcorrections"] == 0]
    if safe:
        best = max(safe, key=lambda r: r["f1"])
        print(f"[운영점 선택] 제약 '유해 오교정 0건' 하에서 개체 F1 최대 → "
              f"자모 {best['jamo']:.2f} / 골격 {best['skeleton']:.2f} (F1 {best['f1']:.3f})")
        out_best = best
    else:
        print("[운영점 선택] 유해 오교정 0건인 설정이 없음 — 임계를 더 올려야 함.")
        out_best = None
    print("[해석] 임계를 낮추면 복원은 늘지만 정상 어절을 성분명으로 끌어간다(없던 약 생성). "
          "임상에선 누락보다 조작이 더 위험하므로 안전 제약을 먼저 걸고 그 안에서 성능을 택한다.")
    return {"rows": out, "operating_point": out_best}


def show_corrections(rows: list[dict], limit: int = 14):
    """실제로 무엇이 무엇으로 복원됐는지 — 감사(audit) 로그."""
    nz = TermNormalizer(all_terms())
    print("\n" + "=" * 92)
    print("교정 감사 로그 — 실제 Whisper 오인식 → 사전 표준형 (샘플)")
    print("=" * 92)
    shown = 0
    for r in rows:
        _, corrs = nz.normalize(_norm_text(r["hyp"]))
        for c in corrs:
            if shown >= limit:
                break
            tag = "자모" if c.layer == "jamo" else "음차골격"
            print(f"  [{r['case_id']:<10}] '{c.span}' → '{c.term}'  ({tag} {c.score:.3f})")
            shown += 1
        if shown >= limit:
            break


def main():
    rows = load_real()
    if rows is None:
        print(f"[!] {REAL} 없음 — 먼저 `python real_asr.py`로 실제 전사를 생성하세요.")
        return
    res = {"real_ablation": {k: v for k, v in real_ablation(rows).items()}}
    show_corrections(rows)
    res["overcorrection"] = overcorrection()
    res["threshold_sweep"] = threshold_sweep(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "normalization.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT / 'normalization.json'}")


if __name__ == "__main__":
    main()
