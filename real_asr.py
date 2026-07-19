"""실제 ASR 측정 — 합성 대화를 TTS로 음성화하고 Whisper로 전사해 *실측* WER·용어 오류율.

결정적 시뮬(run_pipeline.py)이 오류 전파를 *인과적으로* 분리한다면, 여기선 **실제 Whisper가 한국어
진료 대화에서 의료 용어를 얼마나 보존하는가**를 실측해 시뮬의 x축(용어 오류율)을 현실에 정박시킨다.

경로: 정답 대화 → pyttsx3(SAPI5 한국어 Heami) TTS → WAV → faster-whisper 전사 → WER·용어오류율.
- 실 오디오·모델 다운로드 의존이라 **선택적 스크립트**(핵심 실험은 시뮬로 키 없이 재현).
- 정직한 caveat: TTS 합성음은 실제 진료 잡음·억양이 없어 실제 임상 WER의 하한. 그래도 "일반 WER는
  낮은데 의료 용어가 유독 틀린다"는 **방향성**은 실제 모델로도 재현되는지 확인하는 데 의미가 있다.

사용:  pip install faster-whisper pyttsx3  →  python real_asr.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pipeline.asr import term_error_rate, wer
from pipeline.dialogue import CASES, transcript_of

OUT = Path(__file__).resolve().parent / "outputs"
AUDIO = OUT / "audio"


def _norm(s: str) -> str:
    """WER 비교용 정규화 — 문장부호 제거·소문자·공백 단일화(스페이싱 차이 완화)."""
    s = re.sub(r"[^\w가-힣 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def synth_audio(text: str, path: Path) -> bool:
    """pyttsx3(SAPI5)로 한국어 TTS → WAV. 성공 시 True."""
    import pyttsx3
    eng = pyttsx3.init()
    for v in eng.getProperty("voices"):
        if "KO" in v.id.upper() or "KOREAN" in getattr(v, "name", "").upper():
            eng.setProperty("voice", v.id)
            break
    eng.setProperty("rate", 170)
    eng.save_to_file(text, str(path))
    eng.runAndWait()
    return path.exists() and path.stat().st_size > 0


def run(model_size: str = "base") -> dict:
    from faster_whisper import WhisperModel
    AUDIO.mkdir(parents=True, exist_ok=True)
    print(f"[real_asr] Whisper '{model_size}' 로드(첫 실행 시 다운로드)…")
    model = WhisperModel(model_size, device="auto", compute_type="int8")

    rows = []
    for case in CASES:
        clean = transcript_of(case)
        wav = AUDIO / f"{case.case_id}.wav"
        if not wav.exists():
            if not synth_audio(clean, wav):
                print(f"  [skip] TTS 실패: {case.case_id}")
                continue
        segments, _ = model.transcribe(str(wav), language="ko", beam_size=5)
        hyp = " ".join(seg.text.strip() for seg in segments)
        w = wer(_norm(clean), _norm(hyp))
        ter = term_error_rate(case.medical_terms(), _norm(hyp))
        rows.append({"case_id": case.case_id, "wer": round(w, 4), "term_er": round(ter, 4),
                     "hyp": hyp})
        print(f"  {case.case_id:>12} | WER={w:.3f} 용어오류율={ter:.3f}")
        print(f"               ↳ {hyp[:90]}")

    if not rows:
        print("측정된 케이스 없음(TTS/Whisper 확인).")
        return {}
    avg_w = sum(r["wer"] for r in rows) / len(rows)
    avg_t = sum(r["term_er"] for r in rows) / len(rows)
    print(f"\n[실측 요약] 평균 일반 WER={avg_w:.3f} · 평균 의료용어 오류율={avg_t:.3f} "
          f"({len(rows)} 케이스)")
    print(f"[해석] 용어 오류율({avg_t:.3f}) {'>' if avg_t > avg_w else '≈/<'} 일반 WER({avg_w:.3f}) "
          "— 시뮬의 전제(용어에 집중된 실패)가 실제 Whisper로도 " +
          ("재현됨." if avg_t > avg_w else "약하게 나타남(TTS 합성음 한계 가능)."))
    out = {"model": model_size, "avg_wer": round(avg_w, 4), "avg_term_er": round(avg_t, 4),
           "n": len(rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_asr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT / 'real_asr.json'}")
    return out


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "base"
    run(size)
