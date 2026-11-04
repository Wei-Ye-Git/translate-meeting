@echo off
setlocal
cd /d "%~dp0"
set "MEETING_MODE=complete"
call "%~dp0run.bat"
endlocal
