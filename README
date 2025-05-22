# 🤖 Face Recognition with PCA and SVM

This project implements a face recognition system using a combination of Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machines (SVM) for classification. The system is designed to work with the ORL (Olivetti Research Laboratory) Face Database.

## 🌟 Overview

Face recognition is a critical task in various security and biometric identification applications[cite: 6]. This project leverages classical computer vision methods and machine learning techniques to build a robust recognition pipeline. PCA is used to handle the high-dimensional nature of image data by reducing it to a more manageable set of "eigenfaces"[cite: 18, 19], while SVM acts as an efficient classifier in this reduced space[cite: 8].

## ✨ Features

* **ORL Database Loading:** Reads the ORL database containing 40 subjects with 10 images each (totaling 400 images)[cite: 9].
* **Image Preprocessing:** Automatically resizes images to a standard `(112x92)` resolution and converts them to grayscale[cite: 9].
* **PCA Dimensionality Reduction:** Applies PCA to reduce image dimensionality to 50 principal components[cite: 9].
* **SVM Classification:** Trains an SVM classifier with an RBF (Radial Basis Function) kernel using the PCA-transformed image data[cite: 9].
* **Dynamic Test Image Classification:** Allows classification of a dynamically provided test image[cite: 9].
* **Comprehensive Output:**
    * Displays the test image with its predicted class[cite: 9].
    * Shows 9 images from the database belonging to the predicted class[cite: 9].
    * Lists the Top-5 most probable classes with their respective confidences[cite: 9].
* **Cross-Validation & Evaluation:** Performs 5-fold cross-validation and generates a confusion matrix to assess model performance[cite: 11].
* **t-SNE Visualization:** Visualizes the t-SNE projection of PCA vectors, colored by class, to understand data separability[cite: 11].
* **Error Analysis:** Provides insights into frequent errors, separability between subjects, and limitations of the model[cite: 13].

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Ensure you have Python 3.x installed. The project requires the following libraries:

* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `opencv-python` (for `cv2`)
* `Pillow` (for `PIL.Image`)

You can install them using pip:

```bash
pip install numpy matplotlib seaborn scikit-learn opencv-python Pillow
```

### 1. Download the ORL Face Database

The ORL Face Database is publicly available. You'll need to download it and place it in the correct directory.

* **Download Link:** [Original ORL Database Link](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) (or search for "ORL Face Database download")
* **Structure:** After downloading and extracting, you should have a folder (e.g., `orl_faces`) containing 40 subfolders named `s1`, `s2`, ..., `s40`. Each `sX` folder should contain 10 grayscale `.pgm` images (e.g., `1.pgm`, `2.pgm`, ..., `10.pgm`).

**Important:** Place the `orl_faces` directory in the **same directory** as your `app.py` script.

```
your_project_folder/
├── app.py
└── orl_faces/
    ├── s1/
    │   ├── 1.pgm
    │   ├── 2.pgm
    │   └── ...
    ├── s2/
    │   ├── 1.pgm
    │   └── ...
    └── ...
```

### 2. Create a Test Image (Optional but Recommended)

For the `predict_image` function to work with a real image, you can place a test image in the project root directory.

* Name this image `test_image.pgm` (or adjust the `test_image_path` variable in `app.py`).
* Ensure it's a grayscale image.

If `test_image.pgm` is not found, the script will automatically generate a synthetic test image for demonstration purposes.

### 3. Run the Script

Navigate to your project directory in the terminal and run the `app.py` script:

```bash
python app.py
```

The script will perform the following steps:
1.  Load the ORL dataset (or generate synthetic data if not found).
2.  Preprocess the images and apply PCA.
3.  Train the SVM classifier.
4.  Classify the test image (real or synthetic).
5.  Display classification results and visualizations.
6.  Perform cross-validation and show the confusion matrix.
7.  Generate a t-SNE plot.
8.  Provide an analysis of model errors and limitations.

## 📂 Project Structure

```
.
├── app.py                     # Main script containing the face recognition logic
├── IA_AED3_2025_1.pdf         # Project assignment document
├── README.md                  # This README file
└── orl_faces/                 # Directory containing the ORL Face Database
    ├── s1/
    │   ├── 1.pgm
    │   └── ...
    └── ...
```

## 📝 Code Documentation & Analysis

The code (`app.py`) is well-commented to explain each step of the pipeline. The `analyze_errors` method in `app.py` provides an initial discussion on frequent errors, separability, and model limitations, fulfilling a key requirement of the assignment[cite: 13].

## 📚 References

* Kitani, E. C., Thomaz, C. E. (n.d.). *Um Tutorial sobre Análise de Componentes Principais para o Reconhecimento Automático de Faces*. Relatório Técnico Departamento de Engenharia Elétrica-Centro Universitário da FEI. [cite: 18, 19]

---
