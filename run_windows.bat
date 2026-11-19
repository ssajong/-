@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo [DSC PTFE 결정화도 분석기] 필요한 프로그램을 확인하는 중...
python -m pip install -q -r requirements.txt
echo 브라우저에서 분석기가 열립니다. 이 창을 닫으면 프로그램이 종료됩니다.
python -m streamlit run app.py
pause
