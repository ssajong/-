# 🧪 건식전극 PTFE 결정화도 자동 분석 프로그램 (Streamlit)

DSC Raw 데이터(엑셀 `.xlsx` / TRIOS `.txt` / `.csv`)를 업로드하면 PTFE 바인더의
용융 거동(T_onset, T_peak)과 **결정화도 X_c (%)** 를 자동으로 계산합니다.
`PROMPT_DSC_PTFE_Crystallinity_Analyzer.md` 사양을 기본으로 하고, 다중 파일·로트 관리·대시보드를 더했습니다.

## 실행
**Windows**: `run_windows.bat` 더블클릭 (Python 3.10 이상 설치 필요)

**직접 실행**
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```
브라우저가 자동으로 열립니다 (http://localhost:8501).

## 화면 구성
| 위치 | 내용 |
|---|---|
| 사이드바 | 분석 대상 시료 선택, 공정(Mixing/Kneading), PTFE 함량, ΔH₀, 승온 속도, 적분 시작/종료 온도 — 변경 즉시 자동 저장 |
| 📊 데이터 분석 | Raw 파일 업로드(자동 저장) · 🔬 샘플별 분석 리포트 · 📦 로트별 분석 · 🗂️ 전체 데이터 관리(수정·삭제·되돌리기·CSV) |
| 📈 대시보드 | 검색·공정·측정일 필터, 로트/샘플 체크박스 선택, 결정화도 비교(오차막대), 결과 표, 조건 vs X_c, 곡선 겹쳐보기 |
| ⚙️ SOP 설정 | 공정별 적분 구간·표시(미분) 구간·기본 PTFE, 계산 상수, 스무딩, 파일명 해석 규칙 |
| 📖 분석 원리 / 사용법 | 단계별 사용법, 파일명 규칙 + 해석 테스트, 계산식, FAQ |

모든 그래프에 **📋 이미지 복사 / 💾 PNG / 📐 SVG** 버튼이 있습니다.

## 계산식
- ΔHm (J/g) = (60 / β) × ∫ |HF − 직선 베이스라인| dT   (HF: W/g, β: °C/min)
- X_c (%) = ΔHm / (w_PTFE × ΔH₀) × 100   (기본 ΔH₀ = 85.4 J/g)

## 데이터 저장
업로드한 Raw 데이터와 설정은 `app.py` 옆의 `dsc_ptfe_data.db`(SQLite)에 자동 저장됩니다.
이 파일을 복사해 두면 백업이 되며, 다른 PC로 옮겨도 그대로 이어서 사용할 수 있습니다.
