import streamlit as st

# Page setup
trang_chu = st.Page(
    page="views/trang_chu.py",
    title="TRANG CHỦ",
    icon=":material/house:",
    default=True
)
nhan_dien_khuon_mat = st.Page(
    page="views/nhan_dien_khuon_mat.py",
    title="NHẬN DIỆN KHUÔN MẶT",
    icon=":material/account_circle:",
)

nhan_dang_yolo8 = st.Page(
    page="views/nhan_dang_yolo8.py",
    title="NHẬN DẠNG TRÁI CÂY YOLO8",
    icon=":material/nutrition:",
)

xu_ly_anh_so = st.Page(
    page="views/xu_ly_anh_so.py",
    title="XỬ LÝ ẢNH SỐ",
    icon=":material/image:",
)

nhan_dien_ngon_ngu_ky_hieu = st.Page(
    page="views/nhan_dien_ngon_ngu_ky_hieu.py",
    title="NHẬN DIỆN NGÔN NGỮ KÝ HIỆU",
    icon=":material/gesture:",  # icon cho ngôn ngữ ký hiệu, bạn có thể chọn lại nếu thích
)

# Tạo một page cho chức năng "VẼ TRÊN KHÔNG KHÍ"
air_draw = st.Page(
    page="views/ve_tren_khong_khi.py",  # Đường dẫn tới script vẽ ngoài Streamlit
    title="VẼ TRÊN KHÔNG KHÍ",
    icon=":material/brush:",  # Chọn icon cho chức năng vẽ
)

# Navigation
pg = st.navigation(pages=[
    trang_chu,
    nhan_dien_khuon_mat,
    nhan_dang_yolo8,
    xu_ly_anh_so,
    nhan_dien_ngon_ngu_ky_hieu,
    air_draw  # Thêm Air Drawing vào menu điều hướng
])

# Logo
st.logo("assets/logo.png", size="large")

# Run
pg.run()
