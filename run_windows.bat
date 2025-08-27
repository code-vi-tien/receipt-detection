@echo off
title Hệ Thống OCR Hóa Đơn - Bill Detection System
color 0A

echo.
echo     🏦 ================================== 🏦
echo         HỆ THỐNG OCR HÓA ĐƠN
echo         BILL DETECTION SYSTEM
echo     🏦 ================================== 🏦
echo.

echo 🔍 Kiểm tra môi trường...
cd /d "%~dp0"

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python không được tìm thấy!
    echo 📥 Vui lòng cài đặt Python từ: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python đã được cài đặt

REM Kiểm tra virtual environment
if not exist "venv\" (
    echo 🔧 Tạo virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Không thể tạo virtual environment!
        pause
        exit /b 1
    )
)

echo 🚀 Kích hoạt virtual environment...
call venv\Scripts\activate

REM Kiểm tra requirements
if not exist "requirements.txt" (
    echo ⚠️  File requirements.txt không tồn tại!
    echo 📝 Cài đặt dependencies cơ bản...
    pip install torch torchvision ultralytics PySide6 opencv-python paddlepaddle paddleocr numpy Pillow
) else (
    echo 📦 Cài đặt/cập nhật dependencies...
    pip install -r requirements.txt
)

REM Kiểm tra file chính
if not exist "ocr_pipeline_gui.py" (
    echo ❌ File ocr_pipeline_gui.py không tồn tại!
    echo 📁 Vui lòng đảm bảo bạn đang ở đúng thư mục project.
    pause
    exit /b 1
)

echo.
echo 🎉 Khởi động ứng dụng...
echo 💡 Nếu gặp lỗi, vui lòng kiểm tra console log.
echo.

python ocr_pipeline_gui.py

if errorlevel 1 (
    echo.
    echo ❌ Ứng dụng gặp lỗi khi chạy!
    echo 🔧 Vui lòng kiểm tra log ở trên để biết chi tiết.
    echo.
    pause
) else (
    echo.
    echo ✅ Ứng dụng đã đóng thành công.
)

pause
