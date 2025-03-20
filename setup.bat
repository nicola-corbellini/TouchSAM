@echo off

set PYTHON_PATH=""

echo Creating .venv directory...
%PYTHON_PATH% -m venv ".touchsam"
echo Activating virtual environment...
call .touchsam\Scripts\activate
echo Installing dependencies...
echo Installing Torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo Installing Ultralytics
pip install ultralytics==8.3.93
echo Downgrading numpy
pip install numpy<2
echo Installation complete.

pause
