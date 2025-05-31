import streamlit as st
import cv2
import numpy as np
import Chapter.Chapter3 as c3
import Chapter.Chapter4 as c4
import Chapter.Chapter9 as c9


st.set_page_config(page_title="Xử lý ảnh Chương 3 & 4", layout="wide")

# === Danh sách thuật toán theo chương ===
chapter_options = ["Chương 3", "Chương 4", "Chương 9"]

chapter_choice = st.sidebar.selectbox("📘 Chọn chương xử lý ảnh", chapter_options)

if chapter_choice == "Chương 3":
    functions = {
        "1. Negative": ("images/1.tif", c3.Negative),
        "2. Logarit": ("images/2.tif", c3.Logarit),
        "3. Power": ("images/3.tif", c3.Power),
        "4. PiecewiseLinear": ("images/4&6.jpg", c3.PiecewiseLinear),
        "5. Histogram": ("images/5.tif", c3.Histogram),
        "6. Cân bằng Histogram": ("images/4&6.jpg", c3.HistEqual),
        "7. Cân bằng Histogram ảnh màu": ("images/7.tif", c3.HistEqualColor),
        "8. Local Histogram": ("images/8&9.tif", c3.LocalHist),
        "9. Thống kê Histogram": ("images/8&9.tif", c3.HistStat),
        "10. Lọc Gauss (Gaussian Filter)": ("images/10&11.tif", c3.GaussFilter),
        "11. Lọc trung bình (Box Filter)": ("images/10&11.tif", c3.BoxFilter),
        "12. Ngưỡng hóa (Threshold)": ("images/12.tif", c3.Threshold),
        "13. Lọc trung vị (Median Filter)": ("images/13.tif", c3.MedianFilter),
        "14. Làm sắc nét (Sharpen)": ("images/14.tif", c3.Sharpen),
        "15. Gradient": ("images/15.tif", c3.Gradient),
    }

elif chapter_choice == "Chương 4":
    functions = {
        "1. Phổ ảnh (Spectrum)": ("images/c4_1.tif", c4.Spectrum),
        "2. Bộ lọc tần số cao (Frequency Filter)": ("images/10&11.tif", c4.FrequencyFilter),
        "3. Tạo bộ lọc Notch Reject": ("images/c4_1.tif", c4.DrawNotchRejectFilter),
        "4. Khử nhiễu Moire": ("images/c4_2.tif", c4.RemoveMoire),
    }

elif chapter_choice == "Chương 9":
    functions = {
        "1. ConnectedComponent": ("images/ConnectedComponent.tif", c9.ConnectedComponent),
        "2. Đếm hạt gạo": ("images/gao.tif", c9.CountRice),
    }


# === Giao diện chọn thuật toán
choice = st.sidebar.selectbox("🛠️ Chọn thuật toán", list(functions.keys()))
sample_image_path, function_to_apply = functions[choice]

# === Nhận biết ảnh màu
is_color_required = "màu" in choice.lower() or (chapter_choice == "Chương 3" and function_to_apply == c3.HistEqualColor)

# === Đọc ảnh
imgin = cv2.imread(sample_image_path, cv2.IMREAD_COLOR if is_color_required else cv2.IMREAD_GRAYSCALE)

# === Xử lý ảnh
result = function_to_apply(imgin)
if isinstance(result, tuple):
    imgout, info = result
    st.success(f"📊 Kết quả thống kê: {info} đối tượng được phát hiện.")
else:
    imgout = result
    info = None


# === Hiển thị kết quả
col1, col2 = st.columns(2)
with col1:
    st.subheader("🖼️ Ảnh Gốc")
    st.image(imgin, use_container_width=True, channels="GRAY" if len(imgin.shape) == 2 else "RGB")
with col2:
    st.subheader("✅ Ảnh Sau Xử Lý")
    st.image(imgout, use_container_width=True, channels="GRAY" if len(imgout.shape) == 2 else "RGB")

# === Tải ảnh người dùng
st.markdown("---")
uploaded_file = st.file_uploader("📤 Tải ảnh của bạn để thử thuật toán này", type=["jpg", "jpeg", "png", "bmp", "tif"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_custom = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR if is_color_required else cv2.IMREAD_GRAYSCALE)
    img_custom_out = function_to_apply(img_custom)

    st.markdown("### 🔄 Kết quả trên ảnh của bạn:")
    col3, col4 = st.columns(2)
    with col3:
        st.image(img_custom, caption="Ảnh bạn tải lên", use_container_width=True, channels="GRAY" if len(img_custom.shape) == 2 else "RGB")
    with col4:
        st.image(img_custom_out, caption="Ảnh sau khi xử lý", use_container_width=True, channels="GRAY" if len(img_custom_out.shape) == 2 else "RGB")
