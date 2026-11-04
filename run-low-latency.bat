@echo off
setlocal
cd /d "%~dp0"
set "MEETING_MODE=low_latency"
call "%~dp0run.bat"
endlocal
