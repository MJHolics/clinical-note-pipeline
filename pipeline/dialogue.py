"""합성 진료 대화 생성기 — 정답(gold) 구조화 진료기록에서 의료진-환자 대화를 결정적으로 렌더.

실제 진료 음성/텍스트는 개인정보(PHI)라 공개 데이터가 극히 제한적이다(정직한 제약). 그래서
**정답 진료기록(구조화 필드+의료 개체+관계)을 먼저 정의하고 거기서 자연어 대화를 역으로 생성**한다.
이 방향의 이점: 파이프라인(ASR→NER/RE)의 출력과 비교할 **ground truth가 정의상 존재**한다.

각 케이스는 SOAP를 한국어 진료기록에 맞춘 구조화 필드를 가진다:
  - chief_complaint(주소증) · onset(발병 시점) · symptoms(증상 개체) · negated_symptoms(부정 소견)
  - medications(약물 개체) · diagnosis(진단) · plan(계획)
약물은 영문 성분명을 일부러 섞는다 — **ASR이 가장 자주 틀리는 지점**이 의료 용어·영문 약물명이기 때문.
negated_symptoms(pertinent negative)는 대화에서 *명시적으로 부정*되는 증상 — NER이 그냥 문자열 매칭하면
오탐하지만, **부정(negation) 처리를 해야 기록에서 제외**된다(임상적으로 중요: '통증 없음'을 '통증'으로
기록하면 오진).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    case_id: str
    chief_complaint: str
    onset: str                        # 발병 시점(ONSET 관계) 예: "3일 전"
    symptoms: tuple[str, ...]
    negated_symptoms: tuple[str, ...] # 명시적으로 부정된 증상(기록 제외 대상)
    medications: tuple[str, ...]      # 영문 성분명 포함(ASR 난이도)
    diagnosis: str
    plan: tuple[str, ...]

    def medical_terms(self) -> list[str]:
        """이 케이스의 의료 용어(증상·약물·진단) — term-error-rate·개체평가 대상."""
        return [*self.symptoms, *self.medications, self.diagnosis]

    def gold_relations(self) -> set[tuple[str, str, str]]:
        """정답 관계 집합 (head, relation, tail).

        - INDICATES: 각 증상 → 진단(증상이 진단을 시사)
        - TREATS   : 각 약물 → 진단(약물이 진단을 치료)
        - ONSET    : 진단 → 발병시점
        관계 추출(RE)이 대화에서 이 구조를 복원하는지 평가한다.
        """
        rels: set[tuple[str, str, str]] = set()
        for s in self.symptoms:
            rels.add((s, "INDICATES", self.diagnosis))
        for m in self.medications:
            rels.add((m, "TREATS", self.diagnosis))
        if self.onset:
            rels.add((self.diagnosis, "ONSET", self.onset))
        return rels


# 흔한 외래 시나리오 8종(정답 기록). 개체는 한국어 증상 + 영문/한글 약물명 혼합.
# negated_symptoms = 다른 질환의 증상을 pertinent negative로 부정(임상 문진의 감별).
CASES: list[Case] = [
    Case("c1_감기", "기침과 콧물", "3일 전",
         ("기침", "콧물", "인후통", "미열"), ("호흡곤란",),
         ("acetaminophen", "덱스트로메토르판", "슈도에페드린"),
         "급성 상기도 감염", ("수분 섭취", "3일 후 재방문", "해열제 복용")),
    Case("c2_고혈압", "혈압이 높음", "2주 전",
         ("두통", "어지럼", "가슴 답답함"), ("흉통",),
         ("amlodipine", "losartan", "하이드로클로로티아지드"),
         "본태성 고혈압", ("저염식", "매일 혈압 측정", "2주 후 재방문")),
    Case("c3_당뇨", "갈증과 잦은 소변", "한 달 전",
         ("다갈", "다뇨", "체중 감소", "피로감"), ("발열",),
         ("metformin", "글리메피리드"),
         "제2형 당뇨병", ("당화혈색소 검사", "식이 조절", "운동 요법")),
    Case("c4_위염", "명치 통증과 속쓰림", "5일 전",
         ("상복부 통증", "속쓰림", "소화불량", "구역"), ("혈변",),
         ("esomeprazole", "수크랄페이트", "모사프리드"),
         "급성 위염", ("자극적 음식 회피", "금주", "1주 후 재방문")),
    Case("c5_천식", "숨참과 쌕쌕거림", "일주일 전",
         ("호흡곤란", "천명음", "야간 기침"), ("발열",),
         ("salbutamol", "budesonide", "몬테루카스트"),
         "기관지 천식", ("흡입기 사용법 교육", "알레르겐 회피", "폐기능 검사")),
    Case("c6_편두통", "한쪽 머리 통증", "어제",
         ("박동성 두통", "광공포증", "구역", "구토"), ("발열",),
         ("sumatriptan", "나프록센", "메토클로프라미드"),
         "편두통", ("어두운 곳 휴식", "유발 요인 기록", "필요시 진통제")),
    Case("c7_알레르기", "재채기와 눈 가려움", "봄철",
         ("재채기", "비루", "결막 소양감", "코막힘"), ("발열",),
         ("cetirizine", "플루티카손 비강분무", "montelukast"),
         "알레르기 비염", ("항원 회피", "비강 세척", "증상 시 항히스타민제")),
    Case("c8_요로감염", "배뇨통과 빈뇨", "2일 전",
         ("배뇨통", "빈뇨", "잔뇨감", "하복부 불편"), ("측복부 통증",),
         ("ciprofloxacin", "페나조피리딘"),
         "급성 방광염", ("충분한 수분", "소변 배양 검사", "항생제 3일 복용")),
]


def render_dialogue(case: Case) -> list[tuple[str, str]]:
    """정답 케이스 → (화자, 발화) 대화 턴. 결정적(무작위 없음).

    의료진(D)·환자(P) 교대. 의료 개체·관계·부정을 발화 안에 자연스럽게 심는다.
    실제 진료 흐름(주소증→발병→증상 문진→감별 부정→진단 설명→처방)을 모사.
    """
    turns: list[tuple[str, str]] = []
    turns.append(("D", "안녕하세요, 어디가 불편해서 오셨어요?"))
    turns.append(("P", f"{case.onset}부터 {case.chief_complaint} 때문에 왔습니다."))
    turns.append(("D", "증상을 좀 더 자세히 말씀해 주시겠어요?"))
    sx = ", ".join(case.symptoms)
    turns.append(("P", f"{sx} 증상이 있어요."))
    if case.negated_symptoms:                       # pertinent negative(부정 소견)
        neg = ", ".join(case.negated_symptoms)
        turns.append(("P", f"그런데 {neg} 증상은 없습니다."))
    turns.append(("D", f"진찰해 보니 {case.diagnosis}으로 보입니다."))
    meds = ", ".join(case.medications)
    turns.append(("D", f"{meds} 처방해 드리겠습니다."))
    plan = ", ".join(case.plan)
    turns.append(("D", f"{plan} 하시고요."))
    turns.append(("P", "네 알겠습니다, 감사합니다."))
    return turns


def transcript_of(case: Case) -> str:
    """대화를 화자 태그 없는 평문 전사(ASR이 내놓는 형태)로 합친다."""
    return " ".join(utt for _, utt in render_dialogue(case))
