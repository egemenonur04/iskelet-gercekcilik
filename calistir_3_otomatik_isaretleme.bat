@echo off
chcp 65001 >NUL
set PYTHONUTF8=1
cd /d "%~dp0"
rem PyTorch henuz Python 3.15 desteklemiyor: yapay zeka kismi ayri bir Python 3.12 ortaminda calisir.
set VENV=%~dp0.venv_yz
if exist "%VENV%\Scripts\python.exe" goto kurulu
echo [0/4] Python 3.12 hazirlaniyor...
py -3.12 --version >NUL 2>NUL && goto venv
echo Python 3.12 winget ile kuruluyor...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
py -3.12 --version >NUL 2>NUL || goto py_yok
:venv
py -3.12 -m venv "%VENV%" || goto hata
:kurulu
set PY="%VENV%\Scripts\python.exe"
echo [1/4] Yapay zeka paketleri kuruluyor (ilk seferde birkac GB indirir)...
%PY% -m pip install --upgrade pip >NUL
%PY% -c "import torch" >NUL 2>NUL || (%PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto hata)
%PY% -m pip install -r requirements.txt ultralytics || goto hata
%PY% -c "import torch;print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo [2/4] BT olculu sentetik iskelet cizimleri uretiliyor...
%PY% 07_sentetik_veri.py --egitim 6000 --dogrulama 600 || goto hata
echo [3/4] Yer isareti modeli egitiliyor (ekran kartiyla 20-60 dk)...
%PY% 08_nokta_modeli_egit.py || goto hata
echo [4/4] Gorseller otomatik isaretleniyor ve rapor uretiliyor...
%PY% 09_otomatik_isaretle.py || goto hata
%PY% 05_skorla.py --isaret otomatik || goto hata
start "" "%~dp0cikti\rapor.html"
pause
exit /b 0
:py_yok
echo Python 3.12 kurulamadi. https://www.python.org/downloads/windows/ adresinden en son Python 3.12 "Windows installer (64-bit)" dosyasini indirip kur, sonra bu dosyayi tekrar calistir.
pause
exit /b 1
:hata
echo HATA: Yukaridaki mesaji Claude ile paylas.
pause
exit /b 1
