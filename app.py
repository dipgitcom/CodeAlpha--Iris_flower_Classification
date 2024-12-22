import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# StandardScaler for consistent scaling
scaler = StandardScaler()

# Streamlit app interface
def main():
    st.title("Iris Flower Species Prediction")

    st.write(
        "Enter the measurements of an Iris flower and get the predicted species:"
    )

    # Input fields for user
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

    # When the user clicks "Predict"
    if st.button("Predict"):
        # Prepare the input features for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the features
        scaled_features = scaler.fit_transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Map prediction result to the actual class
        species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_map[prediction[0]]

        # Display the result
        st.write(f"Predicted Species: {predicted_species}")

if __name__ == '__main__':
    main()
