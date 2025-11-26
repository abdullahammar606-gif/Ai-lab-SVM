import streamlit as st
import numpy as np
import pickle

# Load the single pickle file
models = pickle.load(open("svm_models.pkl", "rb"))

st.title("Iris Flower Prediction using SVM")
st.write("Select SVM kernel and enter flower measurements.")

# Kernel selection
kernel = st.selectbox(
    "Choose SVM Kernel:",
    ["linear", "rbf", "poly"]
)

model = models[kernel]

# Inputs
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width", 0.0, 10.0, 0.2)

# Predict
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"Predicted Species: **{prediction[0]}**")
