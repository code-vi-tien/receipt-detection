#!/bin/bash

# Hệ Thống OCR Hóa Đơn - Linux Launcher
# Bill Detection System

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "     🏦 ================================== 🏦"
    echo "         HỆ THỐNG OCR HÓA ĐƠN"
    echo "         BILL DETECTION SYSTEM"
    echo "     🏦 ================================== 🏦"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}🔍 $1${NC}"
}

# Main execution
main() {
    clear
    print_header
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    print_info "Kiểm tra môi trường..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 không được tìm thấy!"
        echo -e "${YELLOW}📥 Vui lòng cài đặt Python3:${NC}"
        echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "   CentOS/RHEL:   sudo yum install python3 python3-pip"
        exit 1
    fi
    
    print_success "Python3 đã được cài đặt ($(python3 --version))"
    
    # Check virtual environment
    if [ ! -d "venv" ]; then
        print_info "Tạo virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_error "Không thể tạo virtual environment!"
            echo -e "${YELLOW}💡 Thử cài đặt: sudo apt install python3-venv${NC}"
            exit 1
        fi
        print_success "Virtual environment đã được tạo"
    fi
    
    print_info "Kích hoạt virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Cập nhật pip..."
    python3 -m pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_info "Cài đặt/cập nhật dependencies..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            print_warning "Có lỗi khi cài đặt một số packages"
            print_info "Thử cài đặt dependencies cơ bản..."
            pip install torch torchvision ultralytics PySide6 opencv-python paddlepaddle paddleocr numpy Pillow
        fi
    else
        print_warning "File requirements.txt không tồn tại!"
        print_info "Cài đặt dependencies cơ bản..."
        pip install torch torchvision ultralytics PySide6 opencv-python paddlepaddle paddleocr numpy Pillow
    fi
    
    # Check main application file
    if [ ! -f "ocr_pipeline_gui.py" ]; then
        print_error "File ocr_pipeline_gui.py không tồn tại!"
        echo -e "${YELLOW}📁 Vui lòng đảm bảo bạn đang ở đúng thư mục project.${NC}"
        exit 1
    fi
    
    # Check display for GUI
    if [ -z "$DISPLAY" ]; then
        print_warning "DISPLAY environment variable chưa được set"
        echo -e "${YELLOW}💡 Nếu bạn đang sử dụng SSH, hãy thử: ssh -X username@hostname${NC}"
        echo -e "${YELLOW}💡 Hoặc thiết lập virtual display: export DISPLAY=:0${NC}"
    fi
    
    echo
    print_info "Khởi động ứng dụng..."
    echo -e "${YELLOW}💡 Nếu gặp lỗi, vui lòng kiểm tra console log.${NC}"
    echo
    
    # Run the application
    python3 ocr_pipeline_gui.py
    
    exit_code=$?
    echo
    
    if [ $exit_code -eq 0 ]; then
        print_success "Ứng dụng đã đóng thành công."
    else
        print_error "Ứng dụng gặp lỗi khi chạy! (Exit code: $exit_code)"
        echo -e "${YELLOW}🔧 Vui lòng kiểm tra log ở trên để biết chi tiết.${NC}"
    fi
    
    echo
    echo -e "${CYAN}👋 Cảm ơn bạn đã sử dụng Hệ Thống OCR Hóa Đơn!${NC}"
}

# Trap to handle Ctrl+C
trap 'echo -e "\n${YELLOW}🛑 Ứng dụng bị dừng bởi người dùng${NC}"; exit 130' INT

# Run main function
main "$@"
