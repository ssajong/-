@echo off
cd /d "%~dp0"
echo [DSC PTFE Analyzer] Installing required packages (first run only)...
python -m pip install -q -r requirements.txt
echo Starting... The analyzer will open in your web browser.
echo Closing this window will stop the program.
python -m streamlit run app.py
pause
