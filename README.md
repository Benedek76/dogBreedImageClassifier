# dogBreedImageClassifier

# 🐶 End-to-End Multi-Class Dog Breed Classification / 

At the end of this project, you’ll get to see my treasure, my little dog, Mogyi. Mogyi == 'Nuts in English.'

This project demonstrates how to build an end-to-end multi-class image classifier using TensorFlow and TensorFlow Hub to classify dog breeds. It employs transfer learning to take a pre-trained model and apply it to the dog breed classification problem.

## 📖 Overview

Have you ever seen a dog on the street or in a cafe and wondered what breed it is? This project aims to solve that problem by using machine learning to classify different dog breeds. We use data from the Kaggle Dog Breed Identification competition, which consists of 10,000+ labeled images of 120 different dog breeds.

The problem we're solving is multi-class image classification, which involves assigning one of several labels (breeds) to an input image. In this case, there are 120 different breeds. 

---

### 📂 Dataset
The dataset used in this project comes from Kaggle's [Dog Breed Identification Competition](https://www.kaggle.com/competitions/dog-breed-identification/data). It consists of over 10,000 images of dogs across 120 unique breeds.

---

### 🛠 Workflow

1. **Data Preparation**
   - Download and unzip the dataset.
   - Convert the images and labels into TensorFlow-friendly formats.

2. **Model Selection (Transfer Learning)**
   - Use TensorFlow Hub to load a pre-trained model [`mobilenet_v2_130_224`](https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/130-224-classification/1?tfhub-redirect=true).
   - The pre-trained model is fine-tuned for our dog breed classification task.

3. **Model Training**
   - Use a subset of data for experimentation and validation.
   - Use TensorBoard for tracking training metrics like loss and accuracy.

4. **Evaluation & Testing**
   - Validate the model on validation data and compare predictions with actual labels.
   - Visualize predictions and model performance using custom plotting functions.
   - Use the trained model to make predictions on unseen test data.

5. **Making Predictions**
   - Once trained, the model is capable of classifying new images. It outputs a probability for each breed.

---

### 🏗 Model Architecture

The model uses transfer learning with TensorFlow Hub. It utilizes the following layers:

- **Input Layer**: Images resized to (224, 224).
- **Pre-Trained Model**: `mobilenet_v2_130_224` from TensorFlow Hub.
- **Output Layer**: A Dense layer with 120 output units (one for each breed) and a softmax activation function to output class probabilities.

---

### 🧪 Experimentation & Improvements

The model can be trained on a small subset of data (1000 images) to test the pipeline before scaling up to the entire dataset. After confirming functionality, it can be scaled to train on all 10,000 images, resulting in improved accuracy and better performance on unseen data.

---

### 🚀 Results

With transfer learning, the model achieves over 70% accuracy on the validation dataset after training on just 1000 images. After training on the full dataset, the model reaches even higher accuracy.

---

### 🧠 Features

- **Preprocessing**: Images are converted into numerical tensors for efficient GPU processing.
- **Transfer Learning**: We leverage pre-trained models from TensorFlow Hub to reduce training time and improve performance.
- **Callbacks**: TensorBoard and EarlyStopping callbacks are used to monitor the model's performance and prevent overfitting.
- **Visualization**: The notebook includes custom visualization functions for checking model predictions and comparing them to true labels.
- **Model Saving**: Models are saved and can be reloaded without retraining.
- **Custom Image Prediction**: The model can be used to predict the breed of custom dog images outside the provided dataset.

---

### 📊 Tools and Libraries

- **TensorFlow 2.x**
- **TensorFlow Hub**
- **Pandas & NumPy**
- **Matplotlib** (for visualization)
- **Scikit-learn** (for data splitting)

---

### 🚀 How to Run
1. Clone the Repository   
git clone https://github.com/your-username/dog-breed-classification.git
cd dog-breed-classification

2. Install Dependencies   
pip install -r requirements.txt

3. Download the Dataset from Kaggle   
You can download the dataset from Kaggle's Dog Breed Identification competition and place it in the data directory.

4. Run the Jupyter Notebook   
Open the notebook using Jupyter or Google Colab:
jupyter notebook notebooks/dog-vision.ipynb

6. Train the Model   
Run the cells in the notebook to preprocess data, build the model, and start training.

7. Make Predictions   
Create a my-dog-photos folder and upload the photos of the predicted dog.
Use the provided functions in the notebook to make predictions on the test set or custom images.

---

## 📊 Performance Monitoring
TensorBoard logs are automatically generated during training. To visualize:
tensorboard --logdir logs/tensorboard_logs
