@echo off
set PYTHON_PATH=""

:: Define ANSI Escape Sequences for Colors
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

echo %YELLOW%Upgrading pip...%RESET%
%PYTHON_PATH% -m pip install --upgrade pip

echo %YELLOW%Creating .venv directory...%RESET%
%PYTHON_PATH% -m venv ".touchsam" || (
    echo %RED%Failed to create virtual environment.%RESET%
    pause
    exit /b
)

echo %YELLOW%Activating virtual environment...%RESET%
call .touchsam\Scripts\activate.bat || (
    echo %RED%Failed to activate virtual environment.%RESET%
    pause
    exit /b
)

echo %GREEN%Installing dependencies...%RESET%
echo %BLUE%Installing Torch...%RESET%
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || echo %RED%Torch installation failed%RESET%

echo %BLUE%Installing Ultralytics...%RESET%
python -m pip install ultralytics==8.3.93 || echo %RED%Ultralytics installation failed%RESET%

echo %BLUE%Downgrading numpy...%RESET%
python -m pip install numpy==1.26.4 || echo %RED%Numpy downgrade failed%RESET%

echo %GREEN%Installation complete.%RESET%
pause
