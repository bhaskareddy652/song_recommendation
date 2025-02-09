import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
st.title("Your Personalized Song Recommendations")
st.write(  
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."  
)  
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["csv"])
if uploaded_file is not None:
    try:
        # Specify engine="openpyxl" to avoid format detection issues
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        
        # Show success message and dataset preview
        st.success("Dataset loaded successfully! 🎵")
        st.write("### Dataset Preview:")
        st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
