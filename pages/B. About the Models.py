import streamlit as st
import pandas as pd

st.title("About the Models")
st.write("""
    The models are trained on a comprehensive countrywide accident dataset spanning from February 2016 to the end of March 2023.
    Below are the performance metrics of the models used to predict traffic accident severity based on environmental and road conditions:
    """)
    
st.subheader("Model Performance Metrics:")

data = {
        "Model": ["Random Forest", "AdaBoost", "XGBoost"],
        "Accuracy": [0.9426, 0.9420, 0.9429],
        "Precision": [0.6890, 0.4389, 0.7795],
        "Recall": [0.4351, 0.2514, 0.2906],
        "F1 Score": [0.5051, 0.2453, 0.3119]
    }
df = pd.DataFrame(data)
st.table(df)

st.write("### Click *Predict* on the sidebar to use the predictor.")
