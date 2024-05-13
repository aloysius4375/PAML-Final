import streamlit as st

st.title("Welcome to Your Traffic Accident Risk Predictor")
st.image("media/drive safe.gif", use_column_width=True)  # width align with the page
st.write("""
This web application utilizes advanced predictive models to estimate traffic accident risks based on various user-inputted parameters.
Our project leverages machine learning techniques including Random Forests, XGBoost, and AdaBoost to provide accurate and robust traffic accident risk predictions.
This tool is designed to assist in proactive traffic management and safety planning, making it a valuable asset for public safety and urban planning.
""")
st.write("If you have any questions, check out our [documentation](https://www.overleaf.com) and [GitHub](https://github.com).")
st.write("## Instructions for Use")
st.write("""
- Navigate using the sidebar to switch between Home and Prediction pages.
- On the Prediction page, select the desired model, input relevant parameters, and click 'Predict' to view the risk level.
- The results will be displayed on the same page.
""")
st.write("### Click *Predict* on the sidebar to get started.")
