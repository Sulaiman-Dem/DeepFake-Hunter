import streamlit as st
import librosa
from librosa import feature # Feature Extraction of Audio
import joblib
from tensorflow.keras.models import load_model           # To load the model.
from sklearn.preprocessing import StandardScaler # To perform standardization by centering and scaling.
import numpy as np                                       # Data wrangling.
import pandas as pd                                      #To create/manipulate a dataframe


st.set_page_config(
    page_title="Deepfake Hunter",
)

# This is used to load the model whenever necessary.
file_path_model = 'models/model.21-main12345' ### Change link for model

model = load_model(file_path_model)

################################## Feature Extraction for the Audio ##################################

def mel_frequency_cepstral_coefficients(data, sampling_rate, frame_length = 2048, hop_length = 512, flatten:bool = True):
    mfcc = librosa.feature.mfcc(y = data,sr = sampling_rate)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def spectral_centroid(data, sampling_rate, frame_length = 2048, hop_length = 512):
    scentroid = librosa.feature.spectral_centroid(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(scentroid)

def spectral_bandwidth(data, sampling_rate, frame_length = 2048, hop_length = 512):
    sbandwidth = librosa.feature.spectral_bandwidth(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(sbandwidth)

def spectral_rolloff(data, sampling_rate, frame_length = 2048, hop_length = 512):
    srolloff = librosa.feature.spectral_rolloff(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(srolloff)

def spectral_flux(data, sampling_rate):
    sflux = librosa.onset.onset_strength(y = data, sr = sampling_rate)
    return np.squeeze(sflux)

def feature_extraction(data, sampling_rate, frame_length = 2048, hop_length = 512):
    result = np.array([])
    result = np.hstack((result,
                        mel_frequency_cepstral_coefficients(data, sampling_rate, frame_length, hop_length),
                        spectral_centroid(data, sampling_rate, frame_length, hop_length),
                        spectral_bandwidth(data, sampling_rate, frame_length, hop_length),
                        spectral_rolloff(data, sampling_rate, frame_length, hop_length),
                        spectral_flux(data, sampling_rate)
                     ))
    return result

def get_features(file_path, duration = 2.5, offset = 0.6):
    data, sampling_rate = librosa.load(path = file_path, duration = duration, offset = offset)

    audio_1 = feature_extraction(data, sampling_rate)
    audio = np.array(audio_1)
    
    audio_features = audio

    return audio_features

######################################################################################################

# Header
st.header("Deepfake Hunter - Tensorflow Version")

# Columns
col1, col2 = st.columns([1,1], gap='medium')


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
        audios_feat = get_features(input_audio)
        
        audio_feat_df = pd.DataFrame(audios_feat)
        
        if len(audio_feat_df) < 2592:
            for i in range(2592 - len(audio_feat_df)):
                audio_feat_df.loc[len(audio_feat_df)] = 0
                
        # Puts features into a dataframe
        features_df = pd.DataFrame(audio_feat_df)
            
        full_audio_feature_list = []
        audio_features_flattened = np.array(features_df).flatten()
        full_audio_feature_list.append(audio_features_flattened)
        audio_features = pd.DataFrame(full_audio_feature_list)

        # Loads in Standard Scalar used for training the model
        standard_scaler = joblib.load("models/standard_scaler21-main12345.save")### Change link for scaler
        X = standard_scaler.transform(audio_features)

        # Uses the model to predict whether its a Deepfake or not
        y_pred = model.predict(X)
        # Real if y_pred is close to 1, Deepfake if y_pred is close to 0, and unsure if y_pred is close to 0.5
        # Displays prediction
        print(y_pred)
        if y_pred.round() == 1:
            st.write('Real')
            st.write('Our model is ', round((float(y_pred[0])-0.5)*200, 2), '% sure')
        else:
            st.write('Deepfake')
            st.write('Our model is ', round((abs(float(y_pred[0])-0.5))*200, 2), '% sure')

    if st.button(label='Predict'):
        predictfile()


# Header
st.header("Deepfake Hunter - Pytorch Version")
st.write("Under development")