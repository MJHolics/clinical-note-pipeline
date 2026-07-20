"""ASR 후처리 — 의료 용어 정규화(term normalization).

이 프로젝트가 스스로 지목한 결론("재학습보다 용어 정규화가 비용 대비 효과적인 첫 수")을 실제로
구현해 닫는 층이다. 전사(transcript)를 추출기에 넘기기 *전에* 깨진 의료 용어를 사전의 표준형으로
되돌린다.

**왜 기존 퍼지 매칭으론 부족한가(실측 근거).** extract.py의 퍼지는 *문자 다중집합* 유사도라
한국어 ASR의 지배적 실패 유형인 **자모 1개 치환**을 놓친다. 실제 Whisper 오류 22건 기준:

    배뇨통 → 배뉴통 (중성 ㅛ→ㅠ) : 문자 0.667(임계 0.7 미달, 놓침) · 자모 0.857(복원)
    콧물   → 콩물   (종성 ㅅ→ㅇ) : 문자 0.500(놓침)              · 자모 0.833(복원)
    복원 가능 총계 : 문자 8/22  vs  자모 21/22

**두 개의 직교 층**(오류 유형이 근본적으로 다르므로 하나로 못 덮는다):

  L1 자모(jamo) 정규화 — 한국어 용어용. 음절을 초·중·종성으로 분해해 편집거리를 재면
     '한 음절이 통째로 다르다'가 아니라 '자모 1개가 다르다'로 측정돼 실제 오류 구조와 맞는다.
     사전 큐레이션 불필요(렉시콘만 있으면 동작).

  L2 자음 골격(consonant skeleton) 음차 브리지 — 영문 성분명용. `salbutamol`이 '셀뷰터말'로
     들리면 **스크립트가 달라 자모 유사도조차 0.000**이다(실측). 한글엔 f/v/z 음소가 없어
     p/b/j로, c/q/g는 k로, ㄹ은 l/r 양쪽으로 음차되고 **변이는 모음에 집중**된다 → 모음을 버리고
     자음만 음운 등가류로 접어 비교한다(Soundex 계열 발상).
     실측: salbutamol↔셀뷰터말 골격 `slptml` **완전 일치**, 11건 중 10건 복원.

**결정 기록(왜 이 방식인가).**
  A안 LLM 후보정(전사를 LLM에 넣어 교정) — 커버리지 넓지만 비결정적·키 의존·환각으로 *없던 약을
     지어낼* 위험(임상에선 치명적)이라 검증이 어렵다.
  B안 사전 기반 결정적 정규화(채택) — 사전에 있는 표준형으로만 되돌리므로 **환각이 구조적으로
     불가능**하고, 오프라인·재현 가능하며 단위테스트로 상한을 강제할 수 있다.
  → 안정·검증가능성을 우위에 두고 B를 기본층으로 채택. LLM 교정은 이 층 위의 선택적 2차 게이트로
     남긴다(기본을 안정층으로, 트렌디는 옵션).

**임계값은 추측이 아니라 제약 하 최적화로 정했다.** 임계를 낮추면 복원은 늘지만 정상 어절을
성분명으로 끌어간다(= 없던 약을 기록에 만들어냄, 임상에선 누락보다 위험). 그래서 안전 제약을 먼저
걸고 그 안에서 성능을 택했다:

    제약 = 정상 전사에서 **유해 오교정 0건** → 그 안에서 개체 F1 최대
    결과 = 자모 0.85 / 골격 0.70 (개체 F1 0.768)

초기 추측값(0.75/0.60)은 F1이 더 낮았고(0.771이지만) 유해 오교정 7건이라 **기각**됐다.
`normalize_report.py`가 이 스윕과 선택을 매 실행마다 재현한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── 한글 자모 분해 상수 (U+AC00 조합형 공식) ─────────────────────────
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_HANGUL_BASE, _HANGUL_COUNT = 0xAC00, 11172

_PUNCT_RE = re.compile(r"[,\.\?!·:;\"'()\[\]]")


def decompose(ch: str) -> tuple[str, str, str] | None:
    """한글 음절 1자 → (초성, 중성, 종성). 종성 없으면 ''. 한글이 아니면 None."""
    code = ord(ch) - _HANGUL_BASE
    if not (0 <= code < _HANGUL_COUNT):
        return None
    jong = JONG[code % 28]
    return (CHO[code // 588], JUNG[(code % 588) // 28], "" if jong == " " else jong)


def to_jamo(text: str) -> str:
    """문자열 → 자모 시퀀스. 한글은 분해하고, 비한글(영문·숫자)은 그대로, 공백은 버린다."""
    out: list[str] = []
    for ch in text:
        parts = decompose(ch)
        if parts:
            out.extend(p for p in parts if p)
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    """문자 단위 Levenshtein 거리(순수)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def jamo_similarity(a: str, b: str) -> float:
    """자모 시퀀스 기준 유사도 1 - 편집거리/최대길이 (0~1).

    한국어 ASR 오류는 '음절 통째 교체'가 아니라 **자모 1개 치환**인 경우가 지배적이므로
    (배뇨통→배뉴통 = 중성 ㅛ→ㅠ 하나), 자모 단위로 재야 실제 오류 크기가 반영된다.
    """
    ja, jb = to_jamo(a), to_jamo(b)
    if not ja or not jb:
        return 0.0
    return 1.0 - edit_distance(ja, jb) / max(len(ja), len(jb))


# ── L2 음차 브리지: 자음 골격 ────────────────────────────────────────
# 초성/종성 자모 → 라틴 자음. ㅇ은 초성에서 무음(음가 없음).
_CHO_LAT = {"ㄱ": "k", "ㄲ": "k", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t", "ㄹ": "l", "ㅁ": "m",
            "ㅂ": "p", "ㅃ": "p", "ㅅ": "s", "ㅆ": "s", "ㅇ": "", "ㅈ": "j", "ㅉ": "j",
            "ㅊ": "j", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h"}
_JONG_LAT = {"ㄱ": "k", "ㄲ": "k", "ㄳ": "k", "ㄴ": "n", "ㄵ": "n", "ㄶ": "n", "ㄷ": "t",
             "ㄹ": "l", "ㄺ": "k", "ㄻ": "m", "ㄼ": "l", "ㄽ": "l", "ㄾ": "l", "ㄿ": "p",
             "ㅀ": "l", "ㅁ": "m", "ㅂ": "p", "ㅄ": "p", "ㅅ": "t", "ㅆ": "t", "ㅇ": "ng",
             "ㅈ": "t", "ㅊ": "t", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "t"}

# 라틴 음운 등가류 — 한글에 없는 음소가 무엇으로 음차되는지를 접는다.
#   c/q/g/x → k · r → l(ㄹ은 양쪽) · z → j · f/v/b → p · d → t · 반모음 y/w/h 소거
_LAT_EQ = str.maketrans({"c": "k", "q": "k", "g": "k", "x": "k", "r": "l", "z": "j",
                         "f": "p", "v": "p", "b": "p", "d": "t", "y": "", "w": "", "h": ""})
_VOWELS = frozenset("aeiou")


def _squeeze(s: str) -> str:
    """연속 중복 자음 축약(ll→l) — 음차에서 중복은 보존되지 않는다."""
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def skeleton_latin(s: str) -> str:
    """영문 단어 → 자음 골격(모음 제거 + 음운 등가류 + 중복 축약)."""
    folded = s.lower().translate(_LAT_EQ)
    return _squeeze("".join(c for c in folded if c.isalpha() and c not in _VOWELS))


def skeleton_hangul(s: str) -> str:
    """한글 음차 → 자음 골격. 초성·종성만 라틴으로 옮기고 중성(모음)은 버린다."""
    out: list[str] = []
    for ch in s:
        parts = decompose(ch)
        if not parts:
            continue
        cho, _jung, jong = parts
        out.append(_CHO_LAT[cho])
        if jong:
            out.append(_JONG_LAT[jong])
    return _squeeze("".join(out).translate(_LAT_EQ))


def skeleton_similarity(latin_term: str, hangul_span: str) -> float:
    """영문 성분명 ↔ 한글 음차의 자음 골격 유사도(0~1)."""
    a, b = skeleton_latin(latin_term), skeleton_hangul(hangul_span)
    if not a or not b:
        return 0.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))


# ── 정규화기 ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Correction:
    """적용된 교정 1건 — 감사(audit) 기록용. 임상 시스템은 '무엇을 왜 바꿨는지'가 남아야 한다."""
    span: str          # 원 전사의 구간(ASR이 낸 형태)
    term: str          # 사전 표준형으로 교정된 결과
    score: float       # 유사도
    layer: str         # "jamo" | "skeleton"


def _has_hangul(s: str) -> bool:
    return any(decompose(ch) for ch in s)


# ── 오교정 방지 가드 ─────────────────────────────────────────────────
# 골격이 짧으면(자음 3개 이하) 우연 일치가 급증한다. 실측: '왔습니다'(tspnt)가 budesonide(ptsnt)와
# 0.600으로 맞아버렸다 — 짧은 골격끼리는 편집거리 1~2가 곧 높은 유사도라 변별력이 없다.
_MIN_SKELETON = 4

# 한국어 어미·조사로 끝나는 어절은 **약물명 음차일 수 없다**(약물명은 용언 활용을 하지 않는다).
# 임의 불용어 목록이 아니라 폐쇄 부류(closed class)라는 언어적 성질에 기댄 가드 —
# 실측에서 오교정의 대부분('왔습니다·없습니다·보입니다·처방해'→성분명)이 이 한 줄로 사라진다.
_KOREAN_ENDINGS = (
    "습니다", "ㅂ니다", "십니다", "세요", "어요", "아요", "예요", "에요", "네요", "지요",
    "하고요", "시고요", "겠어요", "가요", "나요", "까요", "는데", "지만", "해서", "해요",
    "으로", "에서", "부터", "까지", "이고", "이며", "하며", "한다", "했다", "된다", "이다",
)


def _is_inflected_korean(span: str) -> bool:
    """구간이 한국어 용언 활용/조사로 끝나는가 — 음차 후보에서 배제하기 위한 판정."""
    last = span.split()[-1] if span.split() else span
    return any(last.endswith(e) for e in _KOREAN_ENDINGS)


class TermNormalizer:
    """사전 기반 결정적 용어 정규화기.

    사전(lexicon)에 있는 표준형으로만 되돌리므로 **없는 개체를 지어낼 수 없다**(환각 구조적 불가).
    한글 용어는 L1 자모 유사도로, 영문 성분명은 L2 자음 골격 음차 브리지로 매칭한다.

    layers: 활성화할 층 — ("jamo",) · ("skeleton",) · ("jamo", "skeleton") 으로 ablation 가능.
    """

    def __init__(self, terms: list[str], jamo_threshold: float = 0.85,
                 skeleton_threshold: float = 0.70, max_span: int = 3,
                 layers: tuple[str, ...] = ("jamo", "skeleton")):
        self.terms = list(terms)
        self.jamo_threshold = jamo_threshold
        self.skeleton_threshold = skeleton_threshold
        self.max_span = max_span
        self.layers = layers
        # 한글 용어는 L1, 영문(비한글) 성분명은 L2 대상 — 스크립트로 분기
        self.hangul_terms = [t for t in self.terms if _has_hangul(t)]
        self.latin_terms = [t for t in self.terms if not _has_hangul(t)]
        self._exact = {t.replace(" ", "") for t in self.terms}

    # ── 후보 점수 ────────────────────────────────────────────────
    def _best_match(self, span: str) -> tuple[str, float, str] | None:
        """이 구간에 가장 잘 맞는 사전 표준형 (term, score, layer). 임계 미달이면 None."""
        best: tuple[str, float, str] | None = None
        if "jamo" in self.layers and _has_hangul(span):
            for t in self.hangul_terms:
                s = jamo_similarity(t, span)
                if s >= self.jamo_threshold and (best is None or s > best[1]):
                    best = (t, s, "jamo")
        if "skeleton" in self.layers and _has_hangul(span) and not _is_inflected_korean(span):
            # 영문 성분명이 한글로 음차된 경우 — 자모로는 0.0이라 골격으로만 이어진다.
            # 용언 활용으로 끝나는 구간은 위에서 이미 배제(약물명은 활용하지 않음).
            sk_span = skeleton_hangul(span)
            if len(sk_span) >= _MIN_SKELETON:
                for t in self.latin_terms:
                    if len(skeleton_latin(t)) < _MIN_SKELETON:
                        continue                   # 너무 짧은 골격은 변별력 없음
                    s = skeleton_similarity(t, span)
                    if s >= self.skeleton_threshold and (best is None or s > best[1]):
                        best = (t, s, "skeleton")
        return best

    def normalize(self, text: str) -> tuple[str, list[Correction]]:
        """전사를 정규화 → (정규화된 전사, 교정 목록).

        어절 n-gram(길이 max_span..1)을 훑어 후보를 모으고, **점수 높은 순·긴 구간 우선**으로
        겹치지 않게 적용한다(longest/best-match-wins). 이미 사전 표준형인 구간은 건드리지 않는다
        — 정상 텍스트 불변(오교정 방지의 1차 방어).
        """
        tokens = text.split()
        stripped = [_PUNCT_RE.sub("", t) for t in tokens]

        candidates: list[tuple[float, int, int, str, str]] = []  # (score, -길이, 시작, term, layer)
        for n in range(self.max_span, 0, -1):
            for i in range(len(tokens) - n + 1):
                span = " ".join(stripped[i:i + n]).strip()
                if not span or span.replace(" ", "") in self._exact:
                    continue                       # 이미 정확한 용어 → 보존
                # 구간 안에 *이미 표준형인* 어절이 섞여 있으면 그 구간은 후보에서 제외한다.
                # 실측 오교정: 'salbutamol budesonide 몬테루카스트' 전체가 montelukast와 골격
                # 1.000으로 맞아, 이미 정확한 약 3개를 1개로 뭉개버렸다. 정상 개체를 삼키는
                # 교정은 복원이 아니라 파괴다.
                if any(tok in self._exact for tok in stripped[i:i + n] if tok):
                    continue
                m = self._best_match(span)
                if m:
                    candidates.append((m[1], -n, i, m[0], m[2]))

        # 점수 desc → 긴 구간 우선 → 앞 위치 우선
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        used: set[int] = set()
        out = list(tokens)
        drop: set[int] = set()
        corrections: list[Correction] = []
        for score, neg_n, i, term, layer in candidates:
            n = -neg_n
            if any(k in used for k in range(i, i + n)):
                continue                           # 이미 교정된 구간과 겹침 → 건너뜀
            span_text = " ".join(stripped[i:i + n])
            tail = _PUNCT_RE.findall(tokens[i + n - 1])   # 구간 끝 문장부호 보존
            out[i] = term + ("".join(tail) if tail else "")
            drop.update(range(i + 1, i + n))
            used.update(range(i, i + n))
            corrections.append(Correction(span=span_text, term=term, score=round(score, 4),
                                          layer=layer))

        normalized = " ".join(tok for k, tok in enumerate(out) if k not in drop)
        return normalized, corrections
