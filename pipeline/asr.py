"""ASR 단계 — WER 지표(순수) + 결정적 오류 주입 시뮬레이터 + 선택적 실제 Whisper.

핵심 통찰(이 프로젝트가 측정하려는 것): 진료 대화 ASR의 실패는 **의료 용어·영문 약물명에 집중**된다.
일반 단어 WER는 낮아도 약물명 하나가 틀리면 진료기록의 핵심 개체가 통째로 날아간다. 그래서
일반 WER와 **의료용어 오류율(term error rate)**을 분리 측정한다.

실제 Whisper는 오디오·모델 다운로드·설치에 의존(무료지만 환경 리스크)하므로 **선택적**(--whisper)으로
두고, 핵심 측정은 **결정적 오류 주입**으로 한다 — 오디오 없이 재현되며, 용어 vs 일반 오류를 *독립적으로
제어*해 "무엇이 진료기록 품질을 떨어뜨리는가"를 인과적으로 분리할 수 있다(무작위 실측으론 어려운 통제).
"""
from __future__ import annotations

import hashlib


# ── 순수 지표 ────────────────────────────────────────────────────────
def _edit_distance(a: list[str], b: list[str]) -> int:
    """단어 단위 Levenshtein 거리(WER 분자)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = 편집거리 / 참조 단어 수."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def term_error_rate(terms: list[str], hypothesis: str) -> float:
    """의료 용어 오류율 = 가설 전사에 온전히 보존되지 못한 용어 비율.

    각 용어(다어절 가능)가 hypothesis에 부분 문자열로 남아있으면 보존, 아니면 오류로 센다.
    일반 WER와 달리 *어느 단어가 틀렸는지*(용어냐 아니냐)에 가중을 둔다.
    """
    if not terms:
        return 0.0
    miss = sum(1 for t in terms if t not in hypothesis)
    return miss / len(terms)


# ── 결정적 오류 주입 시뮬레이터 ──────────────────────────────────────
def _hash01(s: str, salt: str) -> float:
    """문자열→[0,1) 결정적 해시(seed 역할, 무작위성 없이 재현)."""
    h = hashlib.sha1(f"{salt}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _corrupt_token(tok: str, salt: str) -> str:
    """한 단어를 ASR 오류처럼 변형(글자 치환/삭제) — 결정적."""
    if len(tok) <= 1:
        return tok
    r = _hash01(tok, salt + "op")
    pos = int(_hash01(tok, salt + "pos") * len(tok))
    if r < 0.5:                                   # 글자 삭제(발음 흘림)
        return tok[:pos] + tok[pos + 1:]
    chars = list(tok)                             # 글자 치환(유사 발음 오인)
    chars[pos] = "x"
    return "".join(chars)


class NoisyASR:
    """clean 전사에 결정적 오류를 주입 — 일반 단어와 의료 용어에 *다른* 오류율 적용.

    general_er: 일반 단어 오류율 · term_er: 의료 용어 오류율(보통 더 높게 설정 = ASR의 실제 편향).
    terms 목록에 있는 단어를 더 자주 망가뜨려 "용어에 집중된 실패"를 모델링한다.
    """
    def __init__(self, general_er: float = 0.05, term_er: float = 0.3, salt: str = "asr"):
        self.general_er = general_er
        self.term_er = term_er
        self.salt = salt

    def transcribe_text(self, clean: str, terms: list[str] | None = None) -> str:
        terms = terms or []
        term_words = set()
        for t in terms:                           # 다어절 용어를 단어 단위로 펼침
            term_words.update(t.split())
        out = []
        for tok in clean.split():
            is_term = tok in term_words
            er = self.term_er if is_term else self.general_er
            if _hash01(tok, self.salt) < er:
                out.append(_corrupt_token(tok, self.salt))
            else:
                out.append(tok)
        return " ".join(out)


class WhisperASR:
    """선택적 실제 ASR — faster-whisper(설치 시). 오디오 파일 경로를 전사.

    설치: pip install faster-whisper. 모델은 첫 사용 시 다운로드(무료). GPU(4080) 자동 사용.
    핵심 측정은 NoisyASR로 하되, 실제 WER를 원하면 이 백엔드로 오디오를 전사해 wer()로 비교.
    """
    def __init__(self, model_size: str = "base", device: str = "auto"):
        from faster_whisper import WhisperModel   # 지연 임포트(미설치 시 여기서만 실패)
        self.model = WhisperModel(model_size, device=device)

    def transcribe_audio(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text.strip() for seg in segments)
