import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load mô hình ASL đã huấn luyện
model = load_model('./model/asl_vgg16_best_weights.h5')

# Danh sách các nhãn lớp ASL (cập nhật danh sách này nếu cần)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Hàm để nhận diện ngôn ngữ ký hiệu từ camera
def recognize_hand_sign(frame):
    # Tiền xử lý ảnh
    # Chuyển từ BGR sang RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Cân bằng ánh sáng bằng cách sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Giảm nhiễu bằng bộ lọc Gaussian
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Resize ảnh về kích thước (64, 64) mà mô hình yêu cầu
    img = Image.fromarray(img)
    img = img.resize((64, 64))  
    img_array = np.array(img)
    
    # Chuẩn hóa ảnh
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    
    # Dự đoán
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    return labels[predicted_class[0]]

# Streamlit UI setup
st.title("Nhận diện ngôn ngữ ký hiệu ASL từ Camera")

st.image('./images/ASL.jpg', caption="Hướng dẫn ngôn ngữ ký hiệu ASL", use_container_width =True)

# Cấu hình để nhận diện và hiển thị video
run = st.checkbox("Bắt đầu nhận diện từ camera")

# Mở video stream từ camera
if run:
    cap = cv2.VideoCapture(0)

    # Thêm một cửa sổ để hiển thị video
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.write("Không thể mở camera")
            break

        # Điều chỉnh kích thước và vị trí của khung nhận diện
        x, y, w, h = 150, 100, 192, 192  # Điều chỉnh lại vị trí và kích thước của khung
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ khung hình màu xanh lá

        # Cắt phần ảnh trong khung để nhận diện
        cropped_frame = frame[y:y + h, x:x + w]
        
        # Nhận diện ngôn ngữ ký hiệu trong khung
        predicted_sign = recognize_hand_sign(cropped_frame)

        # Hiển thị kết quả nhận diện lên video
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Predict: {predicted_sign}", (x, y - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị video với kết quả nhận diện
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.write("Nhấn vào ô checkbox để bắt đầu nhận diện từ camera.")