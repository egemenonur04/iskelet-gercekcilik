@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py
echo [1/4] Kalan BT maskeleri indiriliyor (inenler atlanir)...
%PY% 01_veri_indir.py || goto hata
echo [2/4] Kemikler yeniden olculuyor...
%PY% 02_bt_olcum.py || goto hata
echo [3/4] Model yeniden egitiliyor...
%PY% 03_model_egit.py || goto hata
echo [4/4] Dogrulama...
%PY% 04_dogrula.py || goto hata
echo Tamam.
pause
exit /b 0
:hata
echo HATA: Yukaridaki mesaji Claude ile paylas.
pause
exit /b 1
