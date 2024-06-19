import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

nn_model = load_model('iris_classification_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

st.title('Iris Flower Classification')

st.write("""
This app uses a Neural Network to classify Iris flowers into three species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica
""")

# Input fields for flower measurements
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)


if st.button('Classify'):
    sample_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = nn_model.predict(sample_input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_species = label_encoder.inverse_transform(predicted_class)
    st.write(f"Predicted species: {predicted_species[0]}")


