import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
st.title("🎶 Your Personalized Song Recommendations")
st.write(
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."
)
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Convert release date to year if present
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
    return df
# Preprocessing data to scale the features
def preprocess_data(df):
    required_columns = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                        'acousticness', 'speechiness', 'popularity', 'mode', 'key', 'name'] 
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, None, f"Dataset is missing required columns: {', '.join(missing_cols)}"
    scaler = MinMaxScaler()
    numerical_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                      'acousticness', 'speechiness', 'popularity']
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler, None
# Training the KNN model
def train_model(df):
    feature_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                    'acousticness', 'speechiness', 'popularity', 'mode', 'key']
    nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn_model.fit(df[feature_cols])
    return nn_model
# Function to recommend songs
def recommend_songs(song_name, df, model):
    feature_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                    'acousticness', 'speechiness', 'popularity', 'mode', 'key']
    if song_name not in df['name'].values:
        return "⚠️ Song not found in dataset. Try another song."
    song_index = df[df['name'] == song_name].index[0]
    song_features = df.loc[song_index, feature_cols].values.reshape(1, -1)
    distances, indices = model.kneighbors(song_features)
    recommended_songs = df.iloc[indices[0][1:]]['name'].tolist()
    return recommended_songs
def main():
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df, scaler, error = preprocess_data(df)
        # Error handling for missing data columns
        if error:
            st.error(error)
            return
        nn_model = train_model(df)
        st.header("Dataset Overview")
        st.write("Here are the first few rows of your dataset:")
        st.dataframe(df.head())
        st.header("Find Similar Songs")
        song_name = st.radio("Select a song name:", df['name'].tolist()) 
        if st.button("Recommend"):
            recommendations = recommend_songs(song_name, df, nn_model)
            # Display recommendations or error messages
            if isinstance(recommendations, list):
                st.write("🎶 Recommended Songs:")
                for song in recommendations:
                    st.write(f"- {song}")
            else:
                st.warning(recommendations)
    else:
        st.info("Please upload a CSV file to get started.")
if __name__ == "__main__":
    main()
