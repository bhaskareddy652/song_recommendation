import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
st.title("Your Personalized Song Recommendations")
st.write(  
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."  
)  
file_path = st.text_input("Enter the file path to your dataset (e.g., C:/path/to/your/file.xlsx)")
df = pd.read_excel(file_path)
st.write("Dataset Preview:")
st.dataframe(df.head())
