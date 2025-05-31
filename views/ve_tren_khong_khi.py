import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import views.xu_ly_ban_tay as htp  # Import HandTrack module của bạn
from PIL import Image

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Tạo canvas để vẽ
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Khởi tạo HandTrack detector
detector = htp.handDetector(detectionCon=0.85)

# Vị trí trước đó của ngón tay
prev_x, prev_y = 0, 0
is_drawing = False  # Biến kiểm tra xem có vẽ không

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Không thể mở camera!")
    exit()

# Streamlit interface setup
st.title("Vẽ trên không khí (Air Drawing)")

# Streamlit video component
video_placeholder = st.empty()

# Xóa canvas khi nhấn nút
if st.button("Xóa Canvas"):
    canvas[:] = 0

# Video streaming và vẽ lên canvas
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Không thể đọc khung hình từ camera!")
        break

    # Lật ngược ảnh để không bị ngược
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện tay
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame, draw=False)  # Lấy vị trí landmarks

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # Tọa độ ngón trỏ
        x2, y2 = lmlist[12][1:]  # Tọa độ ngón giữa

        # Kiểm tra nếu ngón tay trỏ đang mở
        fingers = detector.fingersUp()  # Kiểm tra các ngón tay có đang mở không
        if fingers[1] == 1:  # Nếu ngón trỏ mở, bắt đầu vẽ
            if prev_x == 0 and prev_y == 0:  # Lần đầu tiên phát hiện
                prev_x, prev_y = x1, y1
                is_drawing = False  # Không vẽ nếu chưa có ngón tay
            else:
                # Nếu ngón tay di chuyển, bắt đầu vẽ
                if abs(prev_x - x1) > 5 or abs(prev_y - y1) > 5:  # Kiểm tra nếu ngón tay di chuyển đủ xa
                    is_drawing = True

            if is_drawing:
                # Vẽ đường từ điểm trước đó đến điểm hiện tại
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (0, 0, 255), 5)
                prev_x, prev_y = x1, y1
        else:
            # Nếu ngón tay không mở (nắm tay), không vẽ
            prev_x, prev_y = 0, 0
            is_drawing = False

    # Gộp ảnh gốc và canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Convert image to RGB before displaying with Streamlit
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(combined_rgb)
    video_placeholder.image(image, channels="RGB", use_container_width =True)

    # Nhấn phím 'q' để dừng camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()