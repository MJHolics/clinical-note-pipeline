"""E2E 파이프라인 + 오류 전파 실험 — ASR 오류가 진료기록 품질로 얼마나 번지는가.

대화 → (ASR) → 추출 → 평가를 케이스 8종에 대해 돌린다. 핵심 실험 = **의료용어 오류율(term_er)을
스윕**하며 최종 진료기록 개체 F1이 어떻게 무너지는지 측정. 부가로 **퍼지 매칭 추출기**가 경미한 ASR
오류를 얼마나 복원하는지 exact vs fuzzy로 비교한다.

전부 결정적(키·오디오 불필요). 실제 Whisper/Gemini는 각각 --whisper/--gemini 훅으로.

사용:
    python run_pipeline.py            # 오류 전파 스윕(결정적)
    python run_pipeline.py --demo     # 케이스 1건 전 과정 출력(clean→noisy→추출→채점)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pipeline.asr import NoisyASR, term_error_rate, wer
from pipeline.dialogue import CASES, transcript_of
from pipeline.evaluate import aggregate, score_note
from pipeline.extract import RuleExtractor

OUT = Path(__file__).resolve().parent / "outputs"


def run_at_noise(term_er: float, general_er: float, fuzzy: float) -> dict:
    """한 노이즈 수준에서 전 케이스 파이프라인 실행 → 집계 지표 + 실측 WER."""
    asr = NoisyASR(general_er=general_er, term_er=term_er)
    ext = RuleExtractor(fuzzy=fuzzy)
    scores, wers, ters = [], [], []
    for case in CASES:
        clean = transcript_of(case)
        noisy = asr.transcribe_text(clean, terms=case.medical_terms())
        wers.append(wer(clean, noisy))
        ters.append(term_error_rate(case.medical_terms(), noisy))
        pred = ext.extract(noisy)
        scores.append(score_note(case, pred, transcript=noisy))
    agg = aggregate(scores)
    agg["set_term_er"] = term_er
    agg["obs_wer"] = round(sum(wers) / len(wers), 4)
    agg["obs_term_er"] = round(sum(ters) / len(ters), 4)
    agg["fuzzy"] = fuzzy
    return agg


def sweep() -> dict:
    """의료용어 오류율 스윕(0~0.5) × 추출기(exact vs fuzzy) → 오류 전파 곡선."""
    levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows_exact = [run_at_noise(t, general_er=0.05, fuzzy=0.0) for t in levels]
    rows_fuzzy = [run_at_noise(t, general_er=0.05, fuzzy=0.7) for t in levels]
    print("의료용어 오류가 진료기록 품질로 전파되는 정도 (8 케이스 평균)\n")
    print(f"{'term_er':>8} | {'obs_WER':>7} {'obs_TER':>7} | "
          f"{'exact F1':>8} {'약물 recall':>10} | {'fuzzy F1':>8} {'환각율':>6}")
    print("-" * 74)
    for e, fz in zip(rows_exact, rows_fuzzy):
        print(f"{e['set_term_er']:>8.2f} | {e['obs_wer']:>7.3f} {e['obs_term_er']:>7.3f} | "
              f"{e['f1']:>8.3f} {e['med_recall']:>10.3f} | {fz['f1']:>8.3f} {e['hallucination_rate']:>6.3f}")
    # 핵심 수치 요약
    base_f1 = rows_exact[0]["f1"]
    worst = rows_exact[-1]
    fz_gain = rows_fuzzy[-1]["f1"] - rows_exact[-1]["f1"]
    print(f"\n[요약] 무오류 F1={base_f1:.3f} → term_er 0.5서 exact F1={worst['f1']:.3f} "
          f"(약물 recall {worst['med_recall']:.3f}) · 퍼지 복원 이득 +{fz_gain:.3f}")
    print("[해석] ASR 용어 오류가 최종 진료기록 개체를 직접 갉아먹음(전파). 퍼지 매칭은 경미한 오류를"
          " 일부 복원하나 완전 복원은 불가(정직한 절충).")
    return {"levels": levels, "exact": rows_exact, "fuzzy": rows_fuzzy}


def demo():
    """케이스 1건의 전 과정을 눈으로."""
    case = CASES[0]
    clean = transcript_of(case)
    noisy = NoisyASR(general_er=0.05, term_er=0.4).transcribe_text(clean, case.medical_terms())
    print(f"[케이스] {case.case_id}\n")
    print(f"[clean 전사]  {clean}\n")
    print(f"[noisy ASR]   {noisy}\n")
    print(f"[WER] {wer(clean, noisy):.3f}  [의료용어 오류율] {term_error_rate(case.medical_terms(), noisy):.3f}\n")
    for fuzzy in (0.0, 0.7):
        pred = RuleExtractor(fuzzy=fuzzy).extract(noisy)
        sc = score_note(case, pred, transcript=noisy)
        tag = "fuzzy" if fuzzy else "exact"
        print(f"[{tag} 추출] 증상={pred.symptoms} 약물={pred.medications} 진단='{pred.diagnosis}'")
        print(f"          F1={sc['f1']:.3f} 약물recall={sc['med_recall']:.3f} 환각={sc['hallucination_rate']:.3f}\n")


def save(out: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "propagation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT / 'propagation.json'}")
    try:
        _plot(out)
    except Exception as e:
        print(f"[plot skipped] {type(e).__name__}: {e}")


def _plot(out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = _f
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    x = [r["obs_term_er"] for r in out["exact"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, [r["f1"] for r in out["exact"]], "o-", color="#b71c1c", label="exact 추출 F1")
    ax.plot(x, [r["f1"] for r in out["fuzzy"]], "s--", color="#2e7d32", label="fuzzy 추출 F1")
    ax.plot(x, [r["med_recall"] for r in out["exact"]], "^:", color="#1f4e79", label="약물 recall(exact)")
    ax.set_xlabel("관측된 의료용어 오류율 (ASR)")
    ax.set_ylabel("진료기록 개체 F1 / recall")
    ax.set_title("ASR 오류의 진료기록 품질 전파 (8 케이스 평균)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "propagation.png", dpi=130)
    print(f"[saved] {OUT / 'propagation.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    out = sweep()
    save(out)


if __name__ == "__main__":
    main()
