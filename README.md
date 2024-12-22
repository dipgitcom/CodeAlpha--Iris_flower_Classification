# Iris Flower Classification and Prediction Web App 🌸

This project is a web-based application for classifying Iris flower species using a machine learning model. The app is built with **Streamlit** and uses a **Random Forest Classifier** trained on the Iris dataset. Users can input the sepal and petal dimensions to get predictions for the Iris species.

---

## 🚀 Features
- **Interactive Web Interface**: Built with Streamlit for a user-friendly experience.
- **Iris Flower Classification**: Predicts species as `Iris-setosa`, `Iris-versicolor`, or `Iris-virginica`.
- **Machine Learning Model**: Uses a Random Forest Classifier for accurate predictions.
- **Real-time Input**: Accepts dynamic user inputs for sepal and petal dimensions.
- **Visualization**: Displays results interactively with clear predictions.

---

## 🛠️ Tools & Technologies
1. **Python**: Core programming language for the project.
2. **Streamlit**: Framework for building interactive web applications.
3. **Scikit-learn**: Library for machine learning model development.
4. **Pickle**: For saving and loading the trained model.
5. **Pandas**: For data manipulation and preparation.
6. **Numpy**: For numerical computations.

---

## 📊 Dataset
- **Dataset Name**: Iris Flower Dataset
- **Source**: https://github.com/dipgitcom/CodeAlpha--Iris_flower_Classification/blob/main/iris_model.pkl
- **Description**: The dataset consists of 150 samples from three species of Iris flowers:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

## 🧠 Algorithms Used
1. **Random Forest Classifier**:
   - Ensemble learning method.
   - Combines multiple decision trees to improve classification accuracy.

2. **Scaling**:
   - **StandardScaler**: Scales the input data to have a mean of 0 and a standard deviation of 1 for better model performance.

---

## ⚙️ Workflow
1. **Data Loading**:
   - The Iris dataset is loaded using Scikit-learn.
2. **Data Preprocessing**:
   - The features are scaled using StandardScaler.
3. **Model Training**:
   - A Random Forest Classifier is trained on the dataset.
4. **Model Saving**:
   - The trained model is saved using Pickle (`iris_model.pkl`).
5. **Web Application**:
   - A Streamlit app allows users to input flower dimensions and get predictions.

---

## 🖥️ How to Run the Project

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
