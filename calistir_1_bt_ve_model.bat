@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py
echo [1/5] Paketler kuruluyor...
%PY% -m pip install -r requirements.txt || goto hata
echo [2/5] BT maskeleri indiriliyor (Zenodo, TotalSegmentator v2.0.1)...
%PY% 01_veri_indir.py --max-denek 500 || goto hata
echo [3/5] Kemikler olculuyor...
%PY% 02_bt_olcum.py || goto hata
echo [4/5] Model egitiliyor...
%PY% 03_model_egit.py || goto hata
echo [5/5] Sistem BT ile dogrulaniyor...
%PY% 04_dogrula.py || goto hata
echo.
echo Tamam. Sonuclar cikti klasorunde.
start "" "%~dp0cikti"
pause
exit /b 0
:hata
echo.
echo HATA: Yukaridaki mesaji Claude ile paylas.
pause
exit /b 1
