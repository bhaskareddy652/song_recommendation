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

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="odf")  # Alternative for .xls
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")  # For .xlsx


        st.success("File uploaded successfully!")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error loading file: {e}")
