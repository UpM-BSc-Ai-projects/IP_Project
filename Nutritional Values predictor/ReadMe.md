

# Nutritional Value Prediction of Dates Using Machine Learning and Deep Learning

This project aims to predict the nutritional values of dates fruit based on their images using machine learning (ML) and deep learning (DL). The solution leverages Convolutional Neural Networks (CNNs) and pretrained architectures to deliver accurate and efficient predictions.

---

## **Table of Contents**

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Model Architectures](#model-architectures)
5. [Performance](#performance)
6. [Usage](#usage)
7. [Challenges and Future Work](#challenges-and-future-work)
8. [License](#license)

---

## **Objective**

The goal is to estimate the following nutritional parameters of dates fruits:

- Calories
- Carbohydrates
- Proteins
- Total Fat
- Glucose
- Cholesterol
- Vitamins
- Water Content
- Energy

The predictions are made directly from date images, reducing the need for manual or laboratory-based testing.

---

## **Dataset**

1. **Images:**
   - Images of 8 types of dates, stored in separate folders.
   - Each image corresponds to a specific date variety.
2. **Labels:**
   - `nutritional_values.csv`: A file containing nutritional values for each image.

### **Preprocessing**

- Images resized to:
  - **28x28 (CNNs)** for simplicity.
  - **224x224 (Pretrained Models)** for feature extraction.
- Normalized pixel values to range `[0, 1]`.
- Split into **80% training** and **20% testing** datasets.

---

## **Approach**

The project uses two major approaches:

### **1. Machine Learning**

- **Linear Regression**: Baseline model using features extracted from ResNet50.
- **Random Forest Regressor**: Ensemble-based ML model capturing non-linear patterns.

### **2. Deep Learning**

- **Custom CNN Model**: Designed for grayscale 28x28 images.
- **Pretrained Models**: Fine-tuned DL models for advanced predictions:
  - ResNet50
  - InceptionV3
  - DenseNet121 (best performance)

---

## **Model Architectures**

### **Custom CNN**

- Convolutional Layers for feature extraction.
- MaxPooling for dimensionality reduction.
- Dense layers for regression.
- Output: A regression layer predicting 9 nutritional values.

### **Pretrained Models**

- Feature extraction from pretrained layers.
- Custom regression layers:
  - GlobalAveragePooling2D
  - Dense layers for output.

### **Training Parameters**

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 32
- **Epochs:** 20 (with early stopping)

---

## **Performance**

| Model       | Test MSE   | Test MAE   | Remarks                                   |
| ----------- | ---------- | ---------- | ----------------------------------------- |
| Custom CNN  | 0.1190     | 0.2645     | Simple architecture with good results.    |
| DenseNet121 | BEST VALUE | BEST VALUE | Superior accuracy with dense connections. |

---

## **Usage**

### **Requirements**

- Python 3.7+
- TensorFlow, Keras, Pandas, NumPy, Matplotlib, Streamlit

### **Run Instructions**

1. Clone the repository:

   ```bash
   git clone https://github.com/UpM-BSc-Ai-projects/IP_Project.git
   cd nutritional-value-prediction

   2.	Install dependencies:
   ```

pip install -r requirements.txt

    3.	Train the model:

python train_model.py

    4.	Run the Streamlit app:

streamlit run app.py

    5.	Upload a date image to predict its nutritional values.

## Challenges and Future Work

Challenges
• Data Imbalance: Limited samples for certain date types.
• Feature Complexity: Subtle variations in appearance impact predictions.

Future Work
• Expand the dataset for better generalization.
• Implement data augmentation techniques.
• Include uncertainty quantification in predictions.

## License

This project is licensed under the MIT License.

Contributors:
• Abubakar Waziri
• Hamza Alkhaf
• Muhammad Sattar
• Yusef ElNahas
• Ahmad Sulaimani

Contact:
For queries, please email: [4220056@upm.edu.sa]

