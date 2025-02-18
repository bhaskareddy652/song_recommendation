## 🎵 Song Recommendation System
This project is a personalized song recommendation system based on key musical features using K-Nearest Neighbors (KNN). The model finds songs that are most similar to the user's input based on attributes like danceability, energy, valence, and other musical characteristics.

## 🛠️ Technologies Used

Python 

Streamlit for web interface

Pandas for data manipulation

NumPy for numerical operations

Scikit-learn for building the KNN model

Matplotlib for data visualization

## 📊 Data Columns (Example)
The dataset used in the project includes the following features:

valence: The musical positiveness of a song (positive or negative).

year: The release year of the song.

acousticness: The degree of acoustic sound in the song.

artists: The artists performing the song.

danceability: The suitability of the track for dancing.

duration_ms: The duration of the song in milliseconds.

energy: A measure of intensity and activity in the song.

explicit: Whether the song is explicit (1) or clean (0).

id: Unique identifier for the song.

instrumentalness: The amount of instrumental sound in the song.

key: The key of the song.

liveness: A measure of whether the song was recorded in front of a live audience.

loudness: The overall loudness of the song.

mode: The modality of the song (major or minor scale).

name: Name of the song.

popularity: Popularity score of the song.

release_date: The release date of the song.

speechiness: A measure of the presence of spoken words in the song.

tempo: The tempo (speed) of the song.

## 📝 How It Works
# Data Preprocessing: The dataset is first cleaned and preprocessed, scaling numerical columns using MinMaxScaler.
# Model Training: A KNN model is trained using selected features from the dataset to find the most similar songs.
# Recommendation: Once the model is trained, you can enter the name of a song, and the system will recommend similar songs based on their features.
## 🔧 Usage

Upload your dataset: Upload a CSV file with the required song features.

Enter a song: Type the name of the song you want to get recommendations for.

Get recommendations: Click on the "Recommend" button, and the system will display a list of similar songs.
