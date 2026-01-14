@echo off
chcp 65001 >nul
echo ========================================
echo FFmpeg 安装脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 ffmpeg.exe 是否已存在
if exist "tools\ffmpeg\ffmpeg.exe" (
    echo [信息] FFmpeg 已存在，跳过下载
    echo 位置: tools\ffmpeg\ffmpeg.exe
    pause
    exit /b 0
)

echo [信息] 开始下载并安装 FFmpeg...
echo [提示] FFmpeg 文件较大（约 180MB），下载可能需要几分钟
echo.

REM 运行 Python 脚本下载和安装 FFmpeg
python setup_ffmpeg.py
if errorlevel 1 (
    echo [错误] FFmpeg 安装失败，请检查网络连接
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo FFmpeg 已安装到: tools\ffmpeg\ffmpeg.exe
echo.
pause
