import streamlit as st
import librosa
from librosa import feature # Feature Extraction of Audio
import joblib
from tensorflow.keras.models import load_model           # To load the model.
from sklearn.preprocessing import StandardScaler # To perform standardization by centering and scaling.
import numpy as np                                       # Data wrangling.
import pandas as pd                                      #To create/manipulate a dataframe


st.set_page_config(
    page_title="Deepfake Detection",
)

# This is used to load the model whenever necessary.
file_path_model = 'models\model.13_all_data' ### Change link for model

model = load_model(file_path_model)

################################## Feature Extraction for the Audio ##################################

fn_list_i = [
 librosa.feature.melspectrogram,
 librosa.feature.spectral_centroid,
 librosa.feature.spectral_bandwidth,
 librosa.feature.chroma_stft,
 feature.spectral_rolloff,
 librosa.feature.mfcc,
 librosa.onset.onset_strength,
]

fn_list_ii = [
 feature.zero_crossing_rate
]
fn_list_iii = [
 librosa.feature.rms
]

def get_feature_vector(y,sr):
   feat_vect_i = [ np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
   feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii]
   feat_vect_iii = [ np.mean(funct(y=y)) for funct in fn_list_ii]
   feature_vector = feat_vect_i + feat_vect_ii + feat_vect_iii
   return feature_vector

audios_feat = []
def get_feature(file_path):
    y, sr = librosa.load(file_path,sr=None)
    feature_vector = get_feature_vector(y, sr)
    audios_feat.append(feature_vector)

######################################################################################################

# Header
st.header("Deepfake Detector (temp header)")

# Columns
col1, col2 = st.columns([1,1])


# inside of the first column for adding the audio file and displaying it
with col1:

    # Allows the uploading of the audio files
    input_audio = st.file_uploader('Audio File', type=['wav'])

    # display the link to that page.
    st.audio(input_audio)


# inside of column 2 for the results of the detection
with col2:
    def predictfile():
        # Gets feature of the audio file
        get_feature(input_audio)

        # Puts features into a dataframe
        features_df = pd.DataFrame(audios_feat)

        # Loads in Standard Scalar used for training the model
        standard_scaler = joblib.load("models\standard_scaler13_all_data.save")### Change link for scaler
        X = standard_scaler.transform(features_df)

        # Uses the model to predict whether its a Deepfake or not
        y_pred = model.predict(X)
        # Real if y_pred is close to 1, Deepfake if y_pred is close to 0, and unsure if y_pred is close to 0.5
        # Displays prediction
        if y_pred.round() == 1:
            st.write('Real')
            st.write('Our model is ', round((float(y_pred[0])-0.5)*200, 2), '% sure')
        else:
            st.write('Deepfake')
            st.write('Our model is ', round((abs(float(y_pred[0])-0.5))*200, 2), '% sure')

    if st.button(label='Predict'):
        predictfile()
