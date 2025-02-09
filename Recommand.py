import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
st.title("Your Personalized Song Recommendations")
st.write(  
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."  
)  
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel file)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Check file type and read accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)  # Read CSV file
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")  # Read Excel file

        # Show success message and dataset preview
        st.success("Dataset loaded successfully! 🎵")
        st.write("### Dataset Preview:")
        st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
