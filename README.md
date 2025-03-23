## 🎵 Song Recommendation System
This project is a personalized song recommendation system based on key musical features using K-Nearest Neighbors (KNN). The model finds songs that are most similar to the user's input based on attributes like danceability, energy, valence, and other musical characteristics.

## 🛠️ Technologies Used

Python 

1.Streamlit for web interface

2.Pandas for data manipulation

3.NumPy for numerical operations

4.Scikit-learn for building the KNN model

5.Matplotlib for data visualization

## 📊 Data Columns
The dataset used in the project includes the following features:

1.valence: The musical positiveness of a song (positive or negative).

2.year: The release year of the song.

3.acousticness: The degree of acoustic sound in the song.

4.artists: The artists performing the song.

5.danceability: The suitability of the track for dancing.

6.duration_ms: The duration of the song in milliseconds.

7.energy: A measure of intensity and activity in the song.

8.explicit: Whether the song is explicit (1) or clean (0).

9.id: Unique identifier for the song.

10.instrumentalness: The amount of instrumental sound in the song.

12.key: The key of the song.

13.liveness: A measure of whether the song was recorded in front of a live audience.

14.loudness: The overall loudness of the song.

15.mode: The modality of the song (major or minor scale).

16.name: Name of the song.

17.popularity: Popularity score of the song.

18.release_date: The release date of the song.

19.speechiness: A measure of the presence of spoken words in the song.

20.tempo: The tempo (speed) of the song.

## 📝 How It Works
1.Data Preprocessing: The dataset is first cleaned and preprocessed, scaling numerical columns using MinMaxScaler.

2.Model Training: A KNN model is trained using selected features from the dataset to find the most similar songs.

3.Recommendation: Once the model is trained, you can enter the name of a song, and the system will recommend similar songs based on their features.
## 🔧 Usage

1.Upload your dataset: Upload a CSV file with the required song features.

2.Enter a song: Type the name of the song you want to get recommendations for.

3.Get recommendations: Click on the "Recommend" button, and the system will display a list of similar songs.

## Example of usage:

1.Upload a CSV containing songs and their features like danceability, energy, and tempo.

2.Enter a song title (e.g., "Shape of You") and get a list of similar songs!

## 🖥️ Web Interface
The web interface is created using Streamlit, which allows users to upload data and interact with the recommendation model easily.

## Dataset
The dataset can be downloaded from(https://drive.google.com/file/d/1q9TRKQaIStsXpqJcjQ4nrq0EZLy_H2iG/view?usp=drive_link).

