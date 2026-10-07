# 프롬프트: 건식전극 PTFE 결정화도 자동 분석 Streamlit 앱 생성

아래 요구사항을 모두 만족하는 **단일 Python 파일(Streamlit 앱)** 을 작성해줘.

---

## 1. 목적
DSC(시차주사열량계) 장비에서 내보낸 엑셀 Raw 데이터를 업로드하면,
건식전극 내 **PTFE 바인더의 용융 거동(Onset/Peak 온도)과 결정화도(X_c)** 를 자동으로 계산하고
그래프로 보여주는 웹 앱을 만든다.

## 2. 사용 라이브러리
- `io`, `numpy`, `pandas`, `matplotlib.pyplot`, `streamlit`
- `scipy.integrate.trapezoid` (적분), `scipy.signal.savgol_filter` (노이즈 제거)
- 모든 import는 파일 상단에 빠짐없이 작성할 것 (특히 `import streamlit as st` 누락 주의)

## 3. 전역 스타일
- `plt.rcParams`로 폰트 크기 설정: `font.size=10`, `axes.labelsize=11`, `xtick/ytick.labelsize=9`, `legend.fontsize=9`
- `st.set_page_config(page_title='건식전극 PTFE 결정화도 자동 분석기', layout='wide')`
- 제목: `🧪 건식전극 PTFE 결정화도 자동 분석 프로그램`
- 설명 문구: 엑셀(`.xlsx`) Raw 파일의 **116행 이후에서** 'Temperature'와 'Heat Flow' 항목 행을 자동으로 찾아 분석한다는 안내

## 4. 사이드바 (분석 파라미터)
헤더 `⚙️ 분석 파라미터 설정` 아래에 `number_input`으로:

| 항목 | 기본값 | step | 비고 |
|---|---|---|---|
| PTFE 함량 (wt%) | 1.0 | 0.1 | 입력값 ÷ 100 → 분율로 사용 |
| 100% 결정질 PTFE 표준 엔탈피 ΔH₀ (J/g) | 85.4 | 0.1 | |
| 승온 속도 (°C/min) | 10.0 | 1.0 | |

구분선 후 서브헤더 `📐 적분 및 온도 구간 설정`:
- 적분 시작 온도 (°C): 기본 312.0, step 1.0
- 적분 종료 온도 (°C): 기본 354.0, step 1.0

## 5. 파일 업로드 및 헤더 자동 탐지
- `st.file_uploader`로 `.xlsx`, `.xls` 업로드 허용
- 전체 로직은 `try/except`로 감싸고, 예외 시 `st.error('엑셀 파일을 처리하는 도중 오류가 발생했습니다: {e}')` 출력
- 업로드 바이트를 `io.BytesIO`로 감싸 첫 번째 시트를 `header=None`으로 먼저 읽는다.
- **116행(인덱스 115)부터** 아래로 각 행을 문자열(소문자)로 변환해 검사:
  - 온도 키워드: `temp` 또는 `온도`
  - 열흐름 키워드: `heat`, `flow`, `hf`, `열` 중 하나
  - 두 조건을 모두 만족하는 첫 행을 헤더 행으로 지정 (못 찾으면 기본값 인덱스 115)
- 헤더 바로 다음 행은 **단위 행**으로 간주하여 `skiprows`로 제외하고, 해당 헤더로 다시 읽는다.
- 완전히 빈 행은 `dropna(how='all')`로 제거
- 성공 메시지: `파일 업로드 성공! (116행 이후 탐색 결과 -> N행을 헤더로, N+1행을 단위로 인식 완료)` (엑셀 기준 1-based 행 번호 표시)

## 6. 컬럼 선택
- 컬럼명 앞뒤 공백 제거
- 컬럼명에 온도/열흐름 키워드가 포함된 인덱스를 기본값으로 자동 선택 (없으면 0, 1)
- 2열 레이아웃에 `selectbox` 두 개: `온도(Temperature) 컬럼 선택`, `열 플로우(Heat Flow) 컬럼 선택`
- 두 컬럼을 `pd.to_numeric(errors='coerce')`로 변환 후, 둘 다 NaN이 아닌 행만 numpy 배열로 사용
- 유효 행이 15개 미만이면 `st.error`로 컬럼/파일 구조 확인 요청 후 종료

## 7. 분석 알고리즘
0. **진단 정보**: 읽어온 온도 최소~최대 범위와 행 수를 `st.info`로 표시
1. **스무딩 & 미분**
   - Savitzky-Golay 필터: `window_length=15`(데이터가 부족하면 홀수로 보정, 최소 3), `polyorder=3`
   - `np.gradient(smoothed_hf, temp)`로 1차 미분
2. **용융 피크 탐색**
   - 250~400 °C 구간에서 스무딩된 Heat Flow의 **최솟값(argmin, 흡열 피크)** 위치를 `T_peak`로 지정
   - 해당 구간이 없으면 경고(`st.warning`) 후 전체 범위에서 탐색
3. **Onset 온도 (접선 외삽법)**
   - 최저 온도 ~ `T_peak` 구간(데이터 5개 초과 시)에서 미분값이 최소인 점 = 변곡점
   - 베이스라인 y값 = (적분 시작 온도 − 5 °C) ~ 적분 시작 온도 구간의 스무딩 Heat Flow 평균 (없으면 구간 첫 값)
   - `T_onset = T_infl + (baseline_y − h_infl) / slope`
   - 기울기가 0이거나 데이터가 부족하면 `T_onset = T_peak − 10`
4. **엔탈피 적분**
   - 사이드바의 적분 시작~종료 온도 구간 데이터 추출 (포인트 2개 초과 필요)
   - 구간 양 끝점을 잇는 **직선 베이스라인**을 빼고, 차이의 절댓값을 `trapezoid`로 적분 후 **승온 속도로 나눔** → ΔH_m (J/g)
5. **결정화도**
   - `X_c (%) = ΔH_m / (w_PTFE × ΔH₀) × 100` (w_PTFE가 0이면 0)
   - 적분 구간 데이터가 부족하면 `st.error`로 사이드바 온도 범위 확인 요청

## 8. 결과 출력
- 구분선 + 서브헤더 `📊 자동 분석 결과 리포트`
- 3열 `st.metric`: `용융 시작 온도 (T_onset)`, `용융 피크 온도 (T_peak)`, `산출된 결정화도 (X_c)` (모두 소수 둘째 자리)
- `st.info`로 적분 적용 구간과 적용 표준 엔탈피(ΔH₀) 표시

## 9. 그래프 (matplotlib)
- `plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True, dpi=300)`
- **상단**: 스무딩 Heat Flow (파란 실선, lw 1.5)
  - 적분 구간을 주황색 `fill_between`(alpha 0.4)으로 하이라이트
  - `T_peak` 빨간 점선(`--`), `T_onset` 보라 점선(`:`) 수직선, 범례에 온도값 표시
  - y축 `Heat Flow (W/g)`, 제목 `PTFE Melting Behavior & DSC Curve`(굵게), 범례 우상단, 격자(`--`, alpha 0.6)
- **하단**: 1차 미분 (초록 실선, lw 1.5), y=0 회색 점선, y축 범위 −1.0 ~ 1.0
  - x축 `Temperature (°C)`, y축 `Derivative`, 범례 우상단, 격자
- `plt.tight_layout()` 후 `st.pyplot(fig)`로 출력

## 10. 코드 스타일
- 주석과 UI 문구는 한국어, 그래프 라벨은 영어
- 각 단계(1~7)에 번호 주석을 달아 흐름을 알기 쉽게 작성
