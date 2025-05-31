import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import cv2  # Thêm thư viện OpenCV để xử lý màu

# Tiêu đề trang
st.title("🥭 NHẬN DẠNG TRÁI CÂY BẰNG YOLOv8")

# Tải mô hình YOLOv8
@st.cache_resource
def load_model():
    model = YOLO("model/best.pt")  # Đường dẫn tới mô hình đã train
    return model

model = load_model()

# Giao diện tải ảnh
uploaded_file = st.file_uploader("📤 Tải ảnh trái cây để nhận diện", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    # Nhận diện khi người dùng nhấn nút
    if st.button("🔍 Nhận diện"):
        with st.spinner("Đang nhận diện..."):
            # Lưu tạm ảnh
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                img_path = tmp_file.name
                image.save(img_path)

            # Chạy YOLO
            results = model(img_path)

            # Lấy ảnh có kết quả (BGR)
            result_image_bgr = results[0].plot()

            # Chuyển sang RGB để hiển thị đúng màu
            result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)

            # Hiển thị ảnh kết quả
            st.image(result_image_rgb, caption="Ảnh sau khi nhận diện", use_container_width=True)

            # Xóa ảnh tạm
            os.remove(img_path)
