import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
st.title("Your Personalized Song Recommendations")
st.write(  
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."  
)  
st.title("Upload Your Dataset")

uploaded_file = st.file_uploader("Upload your dataset (.csv, .xlsx, or .xls)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Check file extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")  # Ensure xlrd is installed
        else:
            st.error("Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")
        
        st.write("Dataset Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")



