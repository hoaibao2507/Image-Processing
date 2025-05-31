import streamlit as st

import argparse

import numpy as np
import cv2 as cv
import joblib

# Tiêu đề
st.title("🎥 Video Nhận Diện Khuôn Mặt")

# Mô tả
st.markdown("Dưới đây là video demo quá trình nhận diện khuôn mặt:")

# Hiển thị video
# with open("videos/IMG_3210.mp4", "rb") as video_file:
#     video_bytes = video_file.read()
#     st.video(video_bytes)

FRAME_WINDOW = st.image([])

args = {
    'face_detection_model': './model/face_detection_yunet_2023mar.onnx',
    'face_recognition_model': './model/face_recognition_sface_2021dec.onnx',
    'score_threshold': 0.9,
    'nms_threshold': 0.3,
    'top_k': 5000
}


svc = joblib.load('./model/svc.pkl')
mydict = ['Bao','Hao','Long','Nam','Phat']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]),
                         (coords[0] + coords[2], coords[1] + coords[3]),
                         (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


detector = cv.FaceDetectorYN.create(
    args['face_detection_model'],
    "",
    (320, 320),
    args['score_threshold'],
    args['nms_threshold'],
    args['top_k']
)
recognizer = cv.FaceRecognizerSF.create(args['face_recognition_model'],"")

tm = cv.TickMeter()

cap = cv.VideoCapture(0)
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])

dem = 0
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        st.error("❌ Không lấy được hình ảnh từ webcam.")
        break

    # Inference
    tm.start()
    faces = detector.detect(frame)
    tm.stop()

    if faces[1] is not None:
        for face in faces[1]:
            face_align = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            # Lấy nhãn dự đoán
            label_index = test_predict[0]
            result = mydict[label_index]

            coords = face[:-1].astype(np.int32)
            x, y = coords[0], coords[1]
            cv.putText(frame, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    visualize(frame, faces, tm.getFPS())
    FRAME_WINDOW.image(frame, channels='BGR')
cv.destroyAllWindows()