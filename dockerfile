FROM python:3.10-slim-bullseye

# Cài đặt các gói hệ thống cần thiết cho GUI (X11, OpenGL,...)
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    xvfb \
    x11-xserver-utils \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements.txt và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Thiết lập lệnh mặc định khi container khởi chạy
CMD ["python", "gui.py"]