@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py
%PY% 05_skorla.py --isaret insan || goto hata
start "" "%~dp0cikti\rapor.html"
pause
exit /b 0
:hata
echo HATA: Yukaridaki mesaji Claude ile paylas.
pause
exit /b 1
