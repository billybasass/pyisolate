@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo PyIsolate Benchmark Runner for Windows
echo ================================================================
echo.
echo This script will:
echo   1. Check for uv installation
echo   2. Create a virtual environment
echo   3. Install necessary dependencies
echo   4. Run performance and memory benchmarks
echo   5. Collect all results in a single file
echo.
echo ================================================================
echo.

REM Set up paths and filenames
set "SCRIPT_DIR=%~dp0"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_FILE=benchmark_results_%COMPUTERNAME%_%TIMESTAMP%.txt"
set "VENV_DIR=.benchmark_venv"
set "ERROR_LOG=benchmark_errors.log"

REM Clean up any previous error log
if exist "%ERROR_LOG%" del "%ERROR_LOG%"

echo [%date% %time%] Starting benchmark process... > "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo SYSTEM INFORMATION >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo System: Windows >> "%OUTPUT_FILE%"
echo Computer Name: %COMPUTERNAME% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed Windows version
echo Windows Version Details: >> "%OUTPUT_FILE%"
ver >> "%OUTPUT_FILE%" 2>&1
wmic os get Caption,Version,BuildNumber,OSArchitecture,ServicePackMajorVersion /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed CPU information
echo CPU Information: >> "%OUTPUT_FILE%"
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,Architecture /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo Legacy Processor Info: %PROCESSOR_IDENTIFIER% >> "%OUTPUT_FILE%"
echo Number of Processors: %NUMBER_OF_PROCESSORS% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed memory information
echo Memory Information: >> "%OUTPUT_FILE%"
wmic computersystem get TotalPhysicalMemory /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
for /f "tokens=2 delims==" %%a in ('wmic computersystem get TotalPhysicalMemory /format:list ^| findstr "="') do (
    set /a "RAM_GB=%%a/1024/1024/1024" 2>nul
    if not "!RAM_GB!"=="" echo Total RAM: !RAM_GB! GB >> "%OUTPUT_FILE%"
)
echo. >> "%OUTPUT_FILE%"

REM Get detailed video card information
echo Video Card Information: >> "%OUTPUT_FILE%"
wmic path win32_VideoController get Name,AdapterRAM,DriverVersion,DriverDate,VideoProcessor /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
for /f "tokens=2 delims==" %%a in ('wmic path win32_VideoController get AdapterRAM /format:list ^| findstr "AdapterRAM" ^| findstr -v "AdapterRAM=$"') do (
    set /a "VRAM_GB=%%a/1024/1024/1024" 2>nul
    if not "!VRAM_GB!"=="" echo Video RAM: !VRAM_GB! GB >> "%OUTPUT_FILE%"
)
echo. >> "%OUTPUT_FILE%"

REM Get motherboard and system information
echo System Hardware: >> "%OUTPUT_FILE%"
wmic baseboard get Manufacturer,Product,Version /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
wmic computersystem get Manufacturer,Model,SystemType /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Step 1: Check for uv
echo Step 1: Checking for uv installation...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: uv is not installed or not in PATH
    echo.
    echo Please install uv using one of these methods:
    echo.
    echo Option 1: Using PowerShell ^(recommended^):
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Option 2: Using pip:
    echo   pip install uv
    echo.
    echo Option 3: Download from https://github.com/astral-sh/uv/releases
    echo.
    echo After installation, please restart this script.
    echo.
    echo [%date% %time%] ERROR: uv not found >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo uv found: OK
echo [%date% %time%] uv found >> "%OUTPUT_FILE%"

REM Step 2: Create virtual environment
echo.
echo Step 2: Creating virtual environment...
if exist "%VENV_DIR%" (
    echo Removing existing virtual environment...
    rmdir /s /q "%VENV_DIR%" 2>"%ERROR_LOG%"
    if !ERRORLEVEL! NEQ 0 (
        echo WARNING: Could not remove existing venv, continuing anyway...
        type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    )
)

uv venv "%VENV_DIR%" 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    echo Error details:
    type "%ERROR_LOG%"
    echo.
    echo [%date% %time%] ERROR: Failed to create venv >> "%OUTPUT_FILE%"
    type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo Virtual environment created: OK
echo [%date% %time%] Virtual environment created >> "%OUTPUT_FILE%"

REM Step 3: Activate virtual environment
echo.
echo Step 3: Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat" 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    type "%ERROR_LOG%"
    echo [%date% %time%] ERROR: Failed to activate venv >> "%OUTPUT_FILE%"
    type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo Virtual environment activated: OK

REM Step 4: Install pyisolate and dependencies
echo.
echo Step 4: Installing pyisolate and base dependencies...
uv pip install -e ".[bench]" 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install pyisolate
    type "%ERROR_LOG%"
    echo [%date% %time%] ERROR: Failed to install pyisolate >> "%OUTPUT_FILE%"
    type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo pyisolate installed: OK
echo [%date% %time%] pyisolate installed >> "%OUTPUT_FILE%"

REM Step 5: Install PyTorch with correct CUDA version
echo.
echo Step 5: Installing PyTorch with appropriate CUDA support...
echo Running PyTorch installation helper...
python install_torch_cuda.py 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: PyTorch installation helper failed
    echo Attempting manual CPU-only installation...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>"%ERROR_LOG%"
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to install PyTorch
        type "%ERROR_LOG%"
        echo [%date% %time%] ERROR: Failed to install PyTorch >> "%OUTPUT_FILE%"
        type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
        echo.
        echo Continuing without PyTorch - some benchmarks will be skipped
    )
)
echo [%date% %time%] PyTorch installation completed >> "%OUTPUT_FILE%"

REM Verify Python and package versions
echo.
echo Step 6: Verifying installation...
echo. >> "%OUTPUT_FILE%"
echo Package Versions: >> "%OUTPUT_FILE%"
python --version >> "%OUTPUT_FILE%" 2>&1
python -c "import pyisolate; print(f'pyisolate: {pyisolate.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import numpy; print(f'numpy: {numpy.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import torch; print(f'torch: {torch.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import psutil; print(f'psutil: {psutil.__version__}')" >> "%OUTPUT_FILE%" 2>&1
echo. >> "%OUTPUT_FILE%"

REM Step 7: Run performance benchmarks
echo.
echo Step 7: Running performance benchmarks...
echo ================================================================ >> "%OUTPUT_FILE%"
echo PERFORMANCE BENCHMARKS >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo.

cd benchmarks 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: benchmarks directory not found
    echo Make sure you're running this script from the pyisolate root directory
    pause
    exit /b 1
)

echo Running benchmark.py (this may take several minutes)...
python benchmark.py --quick 2>&1 | tee -a "..\%OUTPUT_FILE%"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Performance benchmark failed or was interrupted
    echo [%date% %time%] WARNING: Performance benchmark failed >> "..\%OUTPUT_FILE%"
    echo Error code: %ERRORLEVEL% >> "..\%OUTPUT_FILE%"
    echo. >> "..\%OUTPUT_FILE%"
    echo Continuing with memory benchmarks...
)

REM Step 8: Run memory benchmarks
echo.
echo Step 8: Running memory benchmarks...
echo. >> "..\%OUTPUT_FILE%"
echo ================================================================ >> "..\%OUTPUT_FILE%"
echo MEMORY BENCHMARKS >> "..\%OUTPUT_FILE%"
echo ================================================================ >> "..\%OUTPUT_FILE%"
echo.

echo Running memory_benchmark.py (this may take several minutes)...
python memory_benchmark.py --counts 1,2,5,10 --test-both-modes 2>&1 | tee -a "..\%OUTPUT_FILE%"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Memory benchmark failed or was interrupted
    echo [%date% %time%] WARNING: Memory benchmark failed >> "..\%OUTPUT_FILE%"
    echo Error code: %ERRORLEVEL% >> "..\%OUTPUT_FILE%"
)

cd ..

REM Step 9: Collect additional runtime information
echo.
echo Step 9: Collecting additional runtime information...
echo. >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo RUNTIME INFORMATION >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"

REM Get current memory usage
echo. >> "%OUTPUT_FILE%"
echo Current Memory Usage: >> "%OUTPUT_FILE%"
wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"

REM Try nvidia-smi if available for current GPU status
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo. >> "%OUTPUT_FILE%"
    echo Current NVIDIA GPU Status: >> "%OUTPUT_FILE%"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv >> "%OUTPUT_FILE%" 2>&1
    echo. >> "%OUTPUT_FILE%"
    echo Full nvidia-smi output: >> "%OUTPUT_FILE%"
    nvidia-smi >> "%OUTPUT_FILE%" 2>&1
)

REM Get disk space information
echo. >> "%OUTPUT_FILE%"
echo Disk Space Information: >> "%OUTPUT_FILE%"
wmic logicaldisk get size,freespace,caption /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"

REM Final summary
echo. >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo [%date% %time%] Benchmark collection completed >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"

REM Deactivate virtual environment
call deactivate 2>nul

REM Display completion message
echo.
echo ================================================================
echo BENCHMARK COLLECTION COMPLETED!
echo ================================================================
echo.
echo Results have been saved to: %OUTPUT_FILE%
echo.
echo Please send the file '%OUTPUT_FILE%' back for analysis.
echo.
echo If you encountered any errors, please also include any error
echo messages shown above.
echo.
echo Thank you for running the benchmarks!
echo.
pause
