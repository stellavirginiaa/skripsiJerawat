import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_cropper import st_cropper
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

st.set_page_config(layout="wide")
st.title("Implementasi Algoritma KNN dan Teknik Morfologi Citra dalam Klasifikasi Jenis Jerawat pada Wajah")

# Informasi mengenai jenis jerawat
st.markdown("##### Stella Virginia - 32210065 - SKRIPSI")
st.markdown("### Jenis Jerawat yang Diklasifikasi")
st.write("Berikut adalah tiga jenis jerawat yang dapat dideteksi dan diklasifikasikan oleh sistem:")

# Buat tiga kolom untuk tiga jenis jerawat
col1, col2, col3 = st.columns(3)

with col1:
    st.image("nodule_info.jpg", caption="Nodul", use_container_width=True)
    st.markdown("""
    **Nodul** adalah jerawat besar, dalam, dan menyakitkan yang terbentuk jauh di bawah kulit. Mereka cenderung keras saat disentuh dan bisa meninggalkan bekas luka jika tidak ditangani dengan baik.
    """)

with col2:
    st.image("papule_info.jpg", caption="Papula", use_container_width=True)
    st.markdown("""
    **Papula** adalah benjolan kecil berwarna merah yang menonjol di permukaan kulit.  
    Tidak memiliki kepala putih dan biasanya terasa lunak saat disentuh.
    """)

with col3:
    st.image("pustule_info.jpg", caption="Pustula", use_container_width=True)
    st.markdown("""
    **Pustula** adalah jenis jerawat yang berisi nanah di bagian tengahnya.
    Jerawat ini umumnya memiliki titik putih atau kuning di puncaknya dan dikelilingi oleh area kulit yang meradang berwarna merah.
    """)



# Tambahkan expander di bawah judul
with st.expander("📘 Penjelasan Fitur yang Digunakan dalam Deteksi"):
    st.write("""
    **Fitur yang Diekstrak:**

    1. **Area**: Luas daerah jerawat yang terdeteksi.
    2. **Perimeter**: Panjang garis luar jerawat.
    3. **Major Axis**: Panjang sisi terpanjang dari bentuk jerawat (seperti sumbu panjang elips).
    4. **Minor Axis**: Panjang sisi terpendek dari bentuk jerawat (sumbu pendek elips).
    5. **Convex Area**: Luas dari bentuk cembung yang membungkus jerawat.
    6. **Convex Perimeter**: Panjang garis luar dari bentuk cembung tersebut.
    7. **Roundness**: Seberapa bulat bentuk jerawat.
    8. **Solidity**: Seberapa padat jerawat (dibandingkan dengan area cembungnya).
    9. **Compactness**: Ukuran seberapa rapat atau padat jerawat dilihat dari area dan kelilingnya.
    10. **Eccentricity**: Seberapa lonjong bentuk jerawat (jika makin lonjong, nilainya makin besar).
    11. **Elongation**: Seberapa memanjang bentuk jerawat.
    """)

# ====================== FUNGSI PREPROCESSING ======================
def preprocess_image(img):
    brightness_factor = 0.7
    adjusted_img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(blurred_img)
    _, thresh = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    dilated_img = cv2.dilate(thresh, kernel, iterations=2)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=6)

    contours_normal, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_normal:
        contour = max(contours_normal, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        convex_area = cv2.contourArea(hull)
        total_area = img.shape[0] * img.shape[1]
        if convex_area >= 0.75 * total_area:
            eroded_img = cv2.bitwise_not(eroded_img)

    area = perimeter = major_axis = minor_axis = convex_area = convex_perimeter = 0
    contours_final, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_final:
        contour = max(contours_final, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            _, axes, _ = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
        hull = cv2.convexHull(contour)
        convex_area = cv2.contourArea(hull)
        convex_perimeter = cv2.arcLength(hull, True)

    roundness = (4 * np.pi * area) / (convex_perimeter ** 2) if convex_perimeter != 0 else 0
    solidity = area / convex_area if convex_area != 0 else 0
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    eccentricity = np.sqrt((major_axis ** 2 - minor_axis ** 2) / major_axis ** 2) if major_axis != 0 else 0
    elongation = 1 - (minor_axis / major_axis) if major_axis != 0 else 0

    features = pd.DataFrame({
        'area': [area],
        'perimeter': [perimeter],
        'major_axis': [major_axis],
        'minor_axis': [minor_axis],
        'convex_area': [convex_area],
        'convex_perimeter': [convex_perimeter],
        'roundness': [roundness],
        'solidity': [solidity],
        'compactness': [compactness],
        'eccentricity': [eccentricity],
        'elongation': [elongation]
    })

    img_area = np.zeros_like(adjusted_img)
    if contours_final:
        cv2.drawContours(img_area, [contour], -1, (255, 255, 255), -1)

    return adjusted_img, img_area, features, area

# ====== UPLOAD & INTERACTIVE CROP ======
uploaded = st.file_uploader("📤 Unggah gambar wajah (jpg/png)", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
st.subheader("✏️ Seret dan ubah ukuran kotak crop di bawah ini")

# return_type='both' mengembalikan tuple (PIL.Image, box_dict)
cropped_pil, box = st_cropper(
    img_pil,
    box_color='#00FF00',
    aspect_ratio=(1,1),
    return_type='both',
    key='cropper'
)

# Tampilkan hasil crop
if cropped_pil:
    cropped_np = np.array(cropped_pil)
    cropped = cv2.resize(cropped_np, (350,350))
    st.subheader("📷 Preview Crop (350×350)")
    # st.image(cropped, use_container_width=True)

    # ====== PREPROCESS & EKSTRAKSI ======
    adj, mask, feats, area = preprocess_image(cropped)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(cropped, caption="Original Crop", use_container_width=True)
    with c2:
        st.image(mask, caption=f"Mask Area: {area:.2f}", use_container_width=True)
    with c3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, bm = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(adj, adj, mask=bm)
        st.image(masked, caption="Masked Jerawat", use_container_width=True)

    st.write("🔍 **Fitur yang Diekstrak**")
    st.dataframe(feats)

    # ====== PREDIKSI ======
    scaler = joblib.load("scaler.joblib")
    model  = joblib.load("knn_model.joblib")
    scaled = scaler.transform(feats)
    pred   = model.predict(scaled)[0]
    st.success(f"✅ **Prediksi Kelas Jerawat:** {pred}")
else:
    st.info("👉 Seret kotak hijau untuk memilih area jerawat, lalu lepaskan.")