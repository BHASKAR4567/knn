
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Diamond Price Prediction", page_icon="💎", layout="wide")

@st.cache_resource
def load_model():
    with open("diamond_knn_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("💎 Diamond Price Prediction")
st.write("Predict diamond prices using KNN Regression.")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat", 0.20, 5.00, 1.00)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])

with col2:
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.number_input("Depth (%)", 40.0, 80.0, 61.5)
    table = st.number_input("Table (%)", 40.0, 80.0, 57.0)

with col3:
    x = st.number_input("Length (mm)", 0.1, 15.0, 6.0)
    y = st.number_input("Width (mm)", 0.1, 15.0, 6.0)
    z = st.number_input("Depth (mm)", 0.1, 15.0, 3.7)

if st.button("Predict Diamond Price"):
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_data)

    st.success("Prediction Completed!")
    st.metric("Estimated Diamond Price", f"${prediction[0]:,.2f}")

st.write("Developed using Python, Scikit-learn and Streamlit.")
