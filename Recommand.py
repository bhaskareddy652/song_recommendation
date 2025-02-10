import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
st.title("Your Personalized Song Recommendations")
st.write(  
    "Explore songs similar to your favorite tracks! Based on key musical features, we bring you the best recommendations using KNN."  
)  
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Converts release date to year 
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
    return df
# Preprocess the data
def preprocess_data(df):
    required_columns = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                        'acousticness', 'speechiness', 'popularity', 'mode', 'key', 'name']
    if not all(col in df.columns for col in required_columns):
        return None, None, "Dataset is missing required columns!"
    scaler = MinMaxScaler()
    numerical_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                      'acousticness', 'speechiness', 'popularity']
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler, None
def train_model(df):
    numerical_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                      'acousticness', 'speechiness', 'popularity']
    
    nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn_model.fit(df[numerical_cols + ['mode', 'key']])
    return nn_model
def recommend_songs(song_name, df, model, scaler):
    numerical_cols = ['danceability', 'energy', 'valence', 'liveness', 'tempo', 
                      'acousticness', 'speechiness', 'popularity']
    if song_name not in df['name'].values:
        return "Song not found in dataset. Try another song."
    song_index = df[df['name'] == song_name].index[0]
    song_features = df.loc[song_index, numerical_cols + ['mode', 'key']].values.reshape(1, -1)
    distances, indices = model.kneighbors(song_features)
    recommended_songs = df.iloc[indices[0][1:]]['name'].tolist()
    return recommended_songs
def main():
    st.title("🎵 Song Recommendation System")
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df, scaler, error = preprocess_data(df)
        if error:
            st.error(error)
            return
        nn_model = train_model(df)
        song_name = st.text_input("Enter a song name:", "Paper Doll")
        if st.button("Recommend"):
            recommendations = recommend_songs(song_name, df, nn_model, scaler)
            if isinstance(recommendations, list):
                st.write("🎶 Recommended Songs:")
                for song in recommendations:
                    st.write(f"- {song}")
            else:
                st.warning(recommendations)
if __name__ == "__main__":
    main()
