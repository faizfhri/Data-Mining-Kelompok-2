'''
    Kelompok 2:
    140810220002 - Muhammad Faiz Fahri
    140810220003 - Dylan Amadeus
    140810220007 - Muhammad Zhafran Shiddiq
'''

import streamlit as st
import numpy as np
import cv2
import zipfile
from PIL import Image
from fpdf import FPDF
import io
from sklearn.metrics import silhouette_score

# Kelas untuk KMeans Clustering
class KMeansManual:
    def __init__(self, n_clusters, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    # Inisialisasi pusat klaster secara acak
    def initialize_centroids(self, data):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(data.shape[0])
        centroids = data[random_idx[:self.n_clusters]]
        return centroids

    # Menentukan klaster berdasarkan pusat terdekat
    def assign_clusters(self, data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    # Menghitung ulang pusat klaster sebagai rata-rata dari klaster yang ditentukan
    def compute_centroids(self, data, labels):
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    # Melatih model dengan data
    def fit(self, data):
        self.centroids = self.initialize_centroids(data)

        for i in range(self.max_iters):
            # Menentukan klaster
            self.labels = self.assign_clusters(data, self.centroids)
            
            # Menghitung ulang pusat klaster
            new_centroids = self.compute_centroids(data, self.labels)

            # Memeriksa konvergensi (jika pusat tidak berubah)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    # Memprediksi klaster untuk data baru
    def predict(self, data):
        return self.assign_clusters(data, self.centroids)

# Fungsi untuk menghitung koefisien Silhouette dengan sampling
def calculate_silhouette_score(data, labels, sample_size=10000):
    if len(np.unique(labels)) > 1:  # Memastikan terdapat lebih dari satu klaster
        if len(data) > sample_size:  # Menggunakan sampel data jika terlalu besar
            indices = np.random.choice(len(data), sample_size, replace=False)
            data_sample = data[indices]
            labels_sample = labels[indices]
            return silhouette_score(data_sample, labels_sample)
        else:
            return silhouette_score(data, labels)
    else:
        return None  # Mengembalikan None jika skor Silhouette tidak dapat dihitung

# Fungsi untuk memuat dan memproses gambar
def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Konversi ke format RGB dan normalisasi
    return img_rgb

# Fungsi untuk mengekstrak dan memproses gambar dari file zip
def extract_and_process_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Daftar semua file di dalam zip
        file_list = zip_ref.namelist()
        for file_name in file_list:
            # Memeriksa apakah file adalah gambar berdasarkan ekstensi
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                # Ekstrak file gambar dari zip ke memori
                with zip_ref.open(file_name) as image_file:
                    # Memuat gambar menggunakan OpenCV
                    file_bytes = np.frombuffer(image_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Konversi ke format RGB dan normalisasi
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        images.append(img_rgb)
    return images

# Fungsi untuk memproses satu file gambar (non-zip)
def process_single_image(image_file):
    img = load_image(image_file)
    return img

# Fungsi untuk menyiapkan nilai piksel untuk clustering
def prepare_pixel_values(images):
    pixel_values = [img.reshape((-1, 3)) for img in images]
    pixel_values = np.vstack(pixel_values)  # Menggabungkan semua nilai piksel menjadi satu array
    return np.float32(pixel_values)

# Fungsi untuk melakukan clustering pada gambar
def cluster_image(image, centroids):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Menentukan klaster berdasarkan pusat yang diberikan
    distances = np.sqrt(((pixel_values - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    # Mengganti nilai piksel dengan pusat klaster
    clustered_image = centroids[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)
    return clustered_image, labels

# Fungsi untuk menampilkan analisis hasil clustering
def analyze_clustering_results(silhouette_no_training, silhouette_with_training):
    st.write("## Analisis Hasil Clustering")
    
    # Bandingkan koefisien Silhouette
    st.write("### Koefisien Silhouette")
    st.write(f"- Clustering tanpa data latih: {silhouette_no_training if silhouette_no_training is not None else 'Tidak Tersedia'}")
    st.write(f"- Clustering dengan data latih: {silhouette_with_training if silhouette_with_training is not None else 'Tidak Tersedia'}")
    
    # Interpretasi hasil berdasarkan nilai koefisien Silhouette
    st.write("### Interpretasi Silhouette")
    if silhouette_no_training is not None and silhouette_with_training is not None:
        if silhouette_with_training > silhouette_no_training:
            st.write("Penggunaan data latih meningkatkan hasil clustering, yang terlihat dari nilai koefisien Silhouette yang lebih tinggi.")
        elif silhouette_with_training < silhouette_no_training:
            st.write("Clustering tanpa data latih memberikan hasil yang lebih baik. Penggunaan data latih mungkin menyebabkan overfitting atau kurang cocok untuk pengelompokan.")
        else:
            st.write("Penggunaan data latih dan tanpa data latih memberikan hasil yang sama. Ini mungkin menunjukkan bahwa data latih tidak terlalu mempengaruhi hasil clustering.")
    else:
        st.write("Tidak cukup informasi untuk membandingkan hasil clustering.")
    
    # Diskusi implikasi dari hasil clustering
    st.write("### Implikasi Clustering")
    st.write("""
    Clustering tanpa data latih dapat memberikan hasil yang baik untuk kasus-kasus di mana data latih sulit diperoleh atau tidak tersedia. 
    Namun, menggunakan data latih yang relevan dapat meningkatkan akurasi dan kualitas hasil clustering, terutama untuk aplikasi yang membutuhkan pengelompokan yang lebih spesifik.
    
    Dalam aplikasi nyata, seperti segmentasi gambar, menggunakan data latih yang representatif dari domain tertentu (misalnya, citra medis atau citra satelit) dapat membantu menghasilkan pengelompokan yang lebih bermakna dan dapat ditindaklanjuti.
    """)
    
    # Menyediakan rekomendasi untuk perbaikan hasil clustering
    st.write("### Rekomendasi untuk Peningkatan")
    st.write("""
    - Coba tambahkan lebih banyak data latih yang lebih bervariasi untuk meningkatkan akurasi clustering.
    """)

# Fungsi untuk membuat PDF dengan gambar
def create_pdf(original_image, clustered_image_no_training, clustered_image_with_training):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Menambahkan Gambar Uji Asli
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Original Test Image", ln=True, align='C')
    original_img_path = "original_test_image.png"
    Image.fromarray((original_image * 255).astype(np.uint8)).save(original_img_path)
    pdf.image(original_img_path, x=10, y=30, w=180)

    # Menambahkan Gambar Uji yang Diklaster tanpa Data Latih
    pdf.add_page()
    pdf.cell(200, 10, txt="Clustered Test Image without Training Data (5 clusters)", ln=True, align='C')
    no_training_img_path = "clustered_no_training.png"
    Image.fromarray((clustered_image_no_training * 255).astype(np.uint8)).save(no_training_img_path)
    pdf.image(no_training_img_path, x=10, y=30, w=180)

    # Menambahkan Gambar Uji yang Diklaster dengan Data Latih
    pdf.add_page()
    pdf.cell(200, 10, txt="Clustered Test Image with Training Data (5 clusters)", ln=True, align='C')
    with_training_img_path = "clustered_with_training.png"
    Image.fromarray((clustered_image_with_training * 255).astype(np.uint8)).save(with_training_img_path)
    pdf.image(with_training_img_path, x=10, y=30, w=180)

    # Menyimpan PDF ke bytes
    pdf_output = io.BytesIO()
    pdf.output(dest='S').encode('latin1')  # Menghasilkan PDF sebagai string
    pdf_output.write(pdf.output(dest='S').encode('latin1'))  # Menyimpan string ke BytesIO
    pdf_output.seek(0)  # Menggeser cursor kembali ke awal

    return pdf_output

# Fungsi untuk menampilkan gambar secara berdampingan
def display_side_by_side(image1, image2, caption1="Original", caption2="Clustered"):
    # Memastikan nilai piksel berada dalam rentang [0, 255] dan tipe uint8
    if image1.dtype != np.uint8:
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)
    
    combined_image = np.hstack((image1, image2))  # Menggabungkan gambar secara horizontal
    st.image(combined_image, caption=f"{caption1} | {caption2}", use_column_width=True)

# Fungsi untuk melakukan clustering contoh menggunakan gambar latih
def perform_example_clustering():
    # Gunakan cara yang sama untuk memproses data latih seperti pada bagian input user
    training_image_paths = [
        "data/001.jpg",
        "data/002.jpg",
        "data/003.jpg",
        "data/004.jpg",
        "data/005.jpg"
    ]  # Contoh path untuk gambar latih (dipisah dari contoh gambar uji)
    example_image_path = "data/test.jpg"  # Path untuk contoh gambar uji
    
    # Memuat dan memproses gambar latih
    training_images = []
    for img_path in training_image_paths:
        # Memuat gambar menggunakan OpenCV
        img = cv2.imread(img_path)
        if img is not None:
            # Konversi ke format RGB dan normalisasi
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            training_images.append(img_rgb)

    # Memproses gambar contoh
    example_image = cv2.imread(example_image_path)
    if example_image is not None:
        example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB) / 255.0

    # Menyiapkan nilai piksel untuk clustering, menggunakan preprocessing yang sama
    pixel_values_training = prepare_pixel_values(training_images)

    # Menentukan jumlah klaster, misalnya 5
    num_clusters = 5

    # Melakukan clustering dengan KMeans, menggunakan cara yang sama seperti pada input user
    kmeans_manual = KMeansManual(n_clusters=num_clusters, random_state=42)
    kmeans_manual.fit(pixel_values_training)

    # Melakukan clustering pada gambar contoh menggunakan centroid yang telah dilatih
    clustered_image, _ = cluster_image(example_image, kmeans_manual.centroids)

    # Memastikan hasil clustering berada dalam rentang yang benar
    if clustered_image.dtype != np.uint8:
        clustered_image = np.clip(clustered_image, 0, 1).astype(np.float32)

    return example_image, clustered_image

# Antarmuka Streamlit
st.set_page_config(layout="wide")
st.title("Image Clustering with Training Data")
st.text("Kelompok 2:")
st.text("140810220002 - Muhammad Faiz Fahri")
st.text("140810220003 - Dylan Amadeus")
st.text("140810220007 - Muhammad Zhafran Shiddiq")

# Mengunggah data latih (file zip atau gambar individu)
uploaded_train_files = st.file_uploader("Upload training images (.zip, .jpg, .png, .jpeg)", type=["zip", "jpg", "jpeg", "png"], accept_multiple_files=True)
training_images = []

if uploaded_train_files:
    with st.spinner("Processing training data..."):
        for uploaded_file in uploaded_train_files:
            if uploaded_file.name.endswith(".zip"):
                # Mengekstrak gambar dari file zip termasuk subdirektori
                st.write(f"Processing zip file {uploaded_file.name}...")
                training_images.extend(extract_and_process_images_from_zip(uploaded_file))
            else:
                # Memproses file gambar individu
                training_images.append(process_single_image(uploaded_file))

        st.write(f"Processed {len(training_images)} training images.")

# Menampilkan hasil clustering contoh jika tidak ada gambar uji yang diunggah
uploaded_image = st.file_uploader("Upload a test image (.png, .jpg, .jpeg)", type=["jpg", "png", "jpeg"])
if not uploaded_image:
    st.write("### Example Clustering Result")
    example_image, example_clustered_image = perform_example_clustering()
    display_side_by_side(example_image, example_clustered_image, "Contoh Gambar Asli", "Contoh Hasil Clustering (5 Klaster)")

# Jika gambar uji diunggah, lanjutkan dengan clustering pada gambar yang diunggah
if uploaded_image:
    # Memuat dan menampilkan gambar uji asli
    test_image = load_image(uploaded_image)
    st.image(test_image, caption="Original Test Image", width=600)

    # Memilih jumlah klaster
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=2)

    # Clustering tanpa data latih
    st.write("Clustering without training data...")
    kmeans_manual = KMeansManual(n_clusters=num_clusters, random_state=42)
    kmeans_manual.fit(test_image.reshape(-1, 3).astype(np.float32))
    clustered_image_no_training, labels_no_training = cluster_image(test_image, kmeans_manual.centroids)

    # Memastikan nilai piksel berada dalam rentang yang benar
    if clustered_image_no_training.dtype == np.float32:
        clustered_image_no_training = np.clip(clustered_image_no_training, 0, 1).astype(np.float32)

    # Menampilkan hasil clustering berdampingan
    display_side_by_side(test_image, clustered_image_no_training, "Original Test Image", f"Clustered Test Image without Training Data ({num_clusters} clusters)")

    # Silhouette score untuk clustering tanpa data latih
    silhouette_no_training = calculate_silhouette_score(test_image.reshape(-1, 3).astype(np.float32), labels_no_training, sample_size=30000)
    st.write(f"Silhouette Coefficient (without training data): {silhouette_no_training if silhouette_no_training is not None else 'Not applicable'}")

    # Jika data latih tersedia, lakukan clustering dengan data latih
    if training_images:
        st.write("Clustering with training data...")
        pixel_values_training = prepare_pixel_values(training_images)
        kmeans_training = KMeansManual(n_clusters=num_clusters, random_state=42)
        kmeans_training.fit(pixel_values_training)

        # Clustering gambar uji dengan centroid data latih
        clustered_image_with_training, labels_with_training = cluster_image(test_image, kmeans_training.centroids)

        # Memastikan nilai piksel berada dalam rentang yang benar
        if clustered_image_with_training.dtype == np.float32:
            clustered_image_with_training = np.clip(clustered_image_with_training, 0, 1).astype(np.float32)

        # Menampilkan hasil clustering dengan data latih berdampingan
        display_side_by_side(test_image, clustered_image_with_training, "Original Test Image", f"Clustered Test Image with Training Data ({num_clusters} clusters)")

        # Silhouette score untuk clustering dengan data latih
        silhouette_with_training = calculate_silhouette_score(test_image.reshape(-1, 3).astype(np.float32), labels_with_training, sample_size=30000)
        st.write(f"Silhouette Coefficient (with training data): {silhouette_with_training if silhouette_with_training is not None else 'Not applicable'}")

        analyze_clustering_results(silhouette_no_training, silhouette_with_training)

# Jika ada file data latih dan gambar uji yang diunggah, buat dan unduh PDF
if uploaded_train_files and uploaded_image:
    pdf_output = create_pdf(test_image, clustered_image_no_training, clustered_image_with_training)
    st.download_button("Download PDF", pdf_output, file_name="clustering_results.pdf", mime="application/pdf")