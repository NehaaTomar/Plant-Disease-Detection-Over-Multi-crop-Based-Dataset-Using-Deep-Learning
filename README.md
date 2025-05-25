# Plant Disease Detection Over Multi-Crop Based Dataset Using Deep Learning

A smart and efficient deep learning-based web application to detect plant diseases from leaf images using a CNN model trained on the full 39-class PlantVillage dataset. Designed to assist farmers, agronomists, and researchers with real-time predictions and solutions.

---

## 🌿 Project Overview

This project uses a Convolutional Neural Network (CNN) to classify 39 types of plant diseases across multiple crops (like tomato, potato, corn, apple, etc.). It features a user-friendly Flask web interface where users can upload images of plant leaves and get instant predictions along with treatment suggestions.

---

## 🚀 Key Features

- **Multi-Crop Support**: Classifies diseases across 14+ crops and 39 total classes.
- **Deep Learning Model**: Trained on the full PlantVillage dataset with high accuracy.
- **Web Interface**: Built using Flask for real-time disease detection.
- **Supplementary Information**: CSV-based system offers disease details and cures.
- **Offline Capability**: No internet needed after setup.
- **Git LFS Support**: Handles large model and dataset files efficiently.

---

## 🧠 Model Architecture

- Framework: PyTorch
- Layers:
  - 4 convolutional + pooling layers
  - MaxPooling layers
  - Fully connected Dense layers
- Trained on augmented images from PlantVillage
- Final Model: `plant_disease_model_1_latest.pt` (200MB, handled via Git LFS)

---
## 📁 Project Structure

```

my_project
│
├── app.py                          # Flask web application
├── cnn.py                          # CNN model definition and loading
├── plant_disease_model_1_latest.pt # Trained PyTorch model
├── image_path.zip                  # Leaf images for testing
├── disease_info.csv                # Disease name and description
├── supplement_info.csv             # Treatment and cure suggestions
├── true_labels.csv                 # Labels for evaluation
├── requirements.txt                # Python package requirements
└── README.md                       # Project documentation

````

---

## 📊 Results

- **Overall Accuracy**: 84%+
- **Evaluation Metrics**:
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix
  - ROC-AUC Curves
  - Confusion Matrix: Shows strong classification even for visually similar diseases.
  - Most Accurate Classes: Healthy Tomato, Potato Late Blight, Grape Black Rot
  - Challenging Cases: Tomato Early Blight vs. Septoria Leaf Spot

The model demonstrates excellent generalization across multiple plant species and diseases.

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/NehaaTomar/Plant-Disease-Detection-Over-Multi-crop-Based-Dataset-Using-Deep-Learning.git
cd Plant-Disease-Detection-Over-Multi-crop-Based-Dataset-Using-Deep-Learning
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Start the Web Application

```bash
python app.py
```

### 4. Open in Browser

- Visit: `http://127.0.0.1:5000/`
- Upload a leaf image and view prediction results.
---

## How to Use the Web App
Open the browser at localhost:5000
Upload an image of a plant leaf
- Click Predict
- See:
  - Predicted Crop
  - Detected Disease
  - Treatment or Recommendation (from supplement_info.csv)

---

## ✍️ Authors

* **Neha Tomar**
* **Aayushi Srivastav**
* **Janhavi Tripathi**

---

## ❤️ Acknowledgements

* [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
* PyTorch & Flask open-source communities
* GitHub and Git LFS for handling large files
* OpenAI tools for research and writing assistance

---

## 📝 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute it for academic or non-commercial purposes.
For more details, see the [LICENSE](LICENSE) file in the repository.

---

## 📬 Feedback & Contributions

Found an issue? Want to improve this project?

* Fork the repository
* Create a pull request
* Or open an issue with your suggestion!

Together we can grow healthier crops using AI!

---

> “Early detection of plant disease is crucial for sustainable agriculture and food security.”

```


