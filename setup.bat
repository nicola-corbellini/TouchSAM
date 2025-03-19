@echo off

set PYTHON_PATH=

echo Creating .venv directory...
%PYTHON_PATH% -m venv ".touchsam"
echo Activating virtual environment...
call .touchsam\Scripts\activate
echo Installing dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ultralytics
echo Installation complete.

pause