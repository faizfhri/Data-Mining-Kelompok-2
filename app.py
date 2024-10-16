import streamlit as st
import numpy as np
import cv2
import zipfile
from PIL import Image
from fpdf import FPDF
import io
from sklearn.metrics import silhouette_score

# Class for KMeans Clustering
class KMeansManual:
    def __init__(self, n_clusters, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    # Initialize the cluster centers randomly
    def initialize_centroids(self, data):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(data.shape[0])
        centroids = data[random_idx[:self.n_clusters]]
        return centroids

    # Assign clusters based on closest centroid
    def assign_clusters(self, data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    # Recompute centroids as the mean of the assigned clusters
    def compute_centroids(self, data, labels):
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    # Fit the model to the data
    def fit(self, data):
        self.centroids = self.initialize_centroids(data)

        for i in range(self.max_iters):
            # Assign clusters
            self.labels = self.assign_clusters(data, self.centroids)
            
            # Compute new centroids
            new_centroids = self.compute_centroids(data, self.labels)

            # Check for convergence (if centroids don't change)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    # Predict cluster for new data
    def predict(self, data):
        return self.assign_clusters(data, self.centroids)

# Function to calculate the Silhouette coefficient with sampling
def calculate_silhouette_score(data, labels, sample_size=10000):
    if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
        if len(data) > sample_size:  # Use a sample of the data if it's too large
            indices = np.random.choice(len(data), sample_size, replace=False)
            data_sample = data[indices]
            labels_sample = labels[indices]
            return silhouette_score(data_sample, labels_sample)
        else:
            return silhouette_score(data, labels)
    else:
        return None  # Return None if silhouette score cannot be computed

# Function to load and preprocess the image
def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img_rgb

# Function to extract and process images from a zip file
def extract_and_process_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # List all files in the zip
        file_list = zip_ref.namelist()
        for file_name in file_list:
            # Check if the file is an image based on its extension
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                # Extract image file from the zip to memory
                with zip_ref.open(file_name) as image_file:
                    # Load the image using OpenCV
                    file_bytes = np.frombuffer(image_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Convert to RGB format
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img_rgb)
    return images

# Function to process a single image file (non-zip)
def process_single_image(image_file):
    img = load_image(image_file)
    return img

# Function to prepare pixel values for clustering
def prepare_pixel_values(images):
    pixel_values = [img.reshape((-1, 3)) for img in images]
    pixel_values = np.vstack(pixel_values)  # Combine all pixel values into a single array
    return np.float32(pixel_values)

# Function to cluster an image
def cluster_image(image, centroids):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Assign clusters based on given centroids
    distances = np.sqrt(((pixel_values - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    # Replace pixel values with cluster centroids
    clustered_image = centroids[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)
    return clustered_image, labels

# Function to create PDF with images
def create_pdf(original_image, clustered_image_no_training, clustered_image_with_training):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add Original Test Image
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Original Test Image", ln=True, align='C')
    original_img_path = "original_test_image.png"
    Image.fromarray(original_image).save(original_img_path)
    pdf.image(original_img_path, x=10, y=30, w=180)

    # Add Clustered Test Image without Training Data
    pdf.add_page()
    pdf.cell(200, 10, txt="Clustered Test Image without Training Data (5 clusters)", ln=True, align='C')
    no_training_img_path = "clustered_no_training.png"
    Image.fromarray(clustered_image_no_training).save(no_training_img_path)
    pdf.image(no_training_img_path, x=10, y=30, w=180)

    # Add Clustered Test Image with Training Data
    pdf.add_page()
    pdf.cell(200, 10, txt="Clustered Test Image with Training Data (5 clusters)", ln=True, align='C')
    with_training_img_path = "clustered_with_training.png"
    Image.fromarray(clustered_image_with_training).save(with_training_img_path)
    pdf.image(with_training_img_path, x=10, y=30, w=180)

    # Save PDF to bytes
    pdf_output = io.BytesIO()
    pdf.output(dest='S').encode('latin1')  # Generate the PDF as a string
    pdf_output.write(pdf.output(dest='S').encode('latin1'))  # Save the string to BytesIO
    pdf_output.seek(0)  # Move cursor back to the start

    return pdf_output

# Function to display images side by side
def display_side_by_side(image1, image2, caption1="Original", caption2="Clustered"):
    # Ensure pixel values are in the range [0, 255] and of type uint8
    if image1.dtype != np.uint8:
        image1 = np.clip(image1, 0, 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = np.clip(image2, 0, 255).astype(np.uint8)
    
    combined_image = np.hstack((image1, image2))  # Combine images horizontally
    st.image(combined_image, caption=f"{caption1} | {caption2}", use_column_width=True)

# Function to perform example clustering using training images
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
    
    # Load dan preprocess gambar latih
    training_images = []
    for img_path in training_image_paths:
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is not None:
            # Convert to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            training_images.append(img_rgb)

    # Preprocess example image
    example_image = cv2.imread(example_image_path)
    if example_image is not None:
        example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)

    # Siapkan nilai piksel untuk clustering, menggunakan preprocessing yang sama
    pixel_values_training = prepare_pixel_values(training_images)

    # Tentukan jumlah klaster, misalnya 5
    num_clusters = 5

    # Lakukan clustering dengan KMeans, menggunakan cara yang sama seperti pada input user
    kmeans_manual = KMeansManual(n_clusters=num_clusters, random_state=42)
    kmeans_manual.fit(pixel_values_training)

    # Cluster example image menggunakan centroid yang telah dilatih
    clustered_image, _ = cluster_image(example_image, kmeans_manual.centroids)

    # Pastikan hasil clustering berada dalam rentang yang benar
    if clustered_image.dtype != np.uint8:
        clustered_image = np.clip(clustered_image, 0, 255).astype(np.uint8)

    return example_image, clustered_image

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Image Clustering with Training Data")
st.text("Kelompok 2:")
st.text("140810220002 - Muhammad Faiz Fahri")
st.text("140810220003 - Dylan Amadeus")
st.text("140810220007 - Muhammad Zhafran Shiddiq")

# Upload training data (zip file or individual images)
uploaded_train_files = st.file_uploader("Upload training images (.zip, .jpg, .png, .jpeg)", type=["zip", "jpg", "jpeg", "png"], accept_multiple_files=True)
training_images = []

if uploaded_train_files:
    with st.spinner("Processing training data..."):
        for uploaded_file in uploaded_train_files:
            if uploaded_file.name.endswith(".zip"):
                # Extract images from zip file including subdirectories
                st.write(f"Processing zip file {uploaded_file.name}...")
                training_images.extend(extract_and_process_images_from_zip(uploaded_file))
            else:
                # Process individual image file
                training_images.append(process_single_image(uploaded_file))

        st.write(f"Processed {len(training_images)} training images.")

# Display example clustering result if no test image has been uploaded
uploaded_image = st.file_uploader("Upload a test image (.png, .jpg, .jpeg)", type=["jpg", "png", "jpeg"])
if not uploaded_image:
    st.write("### Example Clustering Result")
    example_image, example_clustered_image = perform_example_clustering()
    display_side_by_side(example_image, example_clustered_image, "Contoh Gambar Asli", "Contoh Hasil Clustering (5 Klaster)")

# If a test image is uploaded, proceed with clustering the uploaded image
if uploaded_image:
    # Load and display the original test image
    test_image = load_image(uploaded_image)
    st.image(test_image, caption="Original Test Image", width=600)

    # Choose the number of clusters
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=2)

    # Cluster without training data
    st.write("Clustering without training data...")
    kmeans_manual = KMeansManual(n_clusters=num_clusters, random_state=42)
    kmeans_manual.fit(test_image.reshape(-1, 3).astype(np.float32))
    clustered_image_no_training, labels_no_training = cluster_image(test_image, kmeans_manual.centroids)

    # Ensure that the pixel values are in the correct range
    if clustered_image_no_training.dtype == np.float32:
        clustered_image_no_training = np.clip(clustered_image_no_training, 0, 255).astype(np.uint8)

    # Display clustered result side by side
    display_side_by_side(test_image, clustered_image_no_training, "Original Test Image", f"Clustered Test Image without Training Data ({num_clusters} clusters)")

    # Silhouette score for clustering without training data
    silhouette_no_training = calculate_silhouette_score(test_image.reshape(-1, 3).astype(np.float32), labels_no_training)
    st.write(f"Silhouette Coefficient (without training data): {silhouette_no_training if silhouette_no_training is not None else 'Not applicable'}")

    # If training data is available, perform clustering with training data
    if training_images:
        st.write("Clustering with training data...")
        pixel_values_training = prepare_pixel_values(training_images)
        kmeans_training = KMeansManual(n_clusters=num_clusters, random_state=42)
        kmeans_training.fit(pixel_values_training)

        # Cluster the test image with training centroids
        clustered_image_with_training, labels_with_training = cluster_image(test_image, kmeans_training.centroids)

        # Ensure that the pixel values are in the correct range
        if clustered_image_with_training.dtype == np.float32:
            clustered_image_with_training = np.clip(clustered_image_with_training, 0, 255).astype(np.uint8)

        # Display clustered result with training data side by side
        display_side_by_side(test_image, clustered_image_with_training, "Original Test Image", f"Clustered Test Image with Training Data ({num_clusters} clusters)")

        # Silhouette score for clustering with training data
        silhouette_with_training = calculate_silhouette_score(test_image.reshape(-1, 3).astype(np.float32), labels_with_training)
        st.write(f"Silhouette Coefficient (with training data): {silhouette_with_training if silhouette_with_training is not None else 'Not applicable'}")

if uploaded_train_files and uploaded_image:
    pdf_output = create_pdf(test_image, clustered_image_no_training, clustered_image_with_training)
    st.download_button("Download PDF", pdf_output, file_name="clustering_results.pdf", mime="application/pdf")