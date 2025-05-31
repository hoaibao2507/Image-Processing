import streamlit as st

st.title("👋 BÁO CÁO CUỐI KỲ MÔN XỬ LÝ ẢNH SỐ")

# Thông tin cá nhân
st.header("👤 Thông tin nhóm")
st.markdown("""
- **Thành viên 1:** Dương Nguyễn Hoài Bảo - 22110283
- **Thành viên 2:** Phạm Quốc Long - 22110366
""")

# Giới thiệu các chức năng
st.header("🧭 Giới thiệu các chức năng")
st.markdown("""
1. **NHẬN DIỆN KHUÔN MẶT** – Nhận diện khuôn mặt từ ảnh hoặc webcam.
2. **NHẬN DẠNG TRÁI CÂY YOLO8** – Phát hiện và phân loại trái cây sử dụng mô hình YOLOv8.
3. **XỬ LÝ ẢNH SỐ** – Các công cụ xử lý ảnh như âm bản, log, histogram,...
4. **NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (ASL - American Sign Language)** – Nhận diện chữ cái ngôn ngữ ký hiệu bằng camera thời gian thực.
5. **VẼ TRÊN KHÔNG KHÍ** – Dùng chuyển động tay để vẽ trong không khí.
""")