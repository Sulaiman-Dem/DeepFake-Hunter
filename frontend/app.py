import streamlit as st
import librosa
from librosa import feature # Feature Extraction of Audio
import joblib
from tensorflow.keras.models import load_model           # To load the model.
from sklearn.preprocessing import StandardScaler # To perform standardization by centering and scaling.
import numpy as np                                       # Data wrangling.
import pandas as pd                                      #To create/manipulate a dataframe
import tempfile #To handle temporary files
from pydub import AudioSegment # To augment audio
from pathlib import Path # File management
from audio_recorder_streamlit import audio_recorder # Voice recorder

uploaded_audio_result = None
preloaded_audio_result = None
voice_recordings_result = None

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
st.write("This is a model that will be able to detect real or fake audio. This current model only accepts wav & mp3 files. You can upload your own audio files or choose between the preloaded audio files.")
st.header("Deepfake Hunter - Tensorflow Version")


# Columns
col1, col2 = st.columns([1,1], gap='medium')

# inside of the first column for adding the audio file and displaying it
with col1:
    # Allows the uploading of the audio files
    input_audio = st.file_uploader('Audio File', type=['wav', 'mp3'])
    # display the link to that page.
    if input_audio is not None:
        st.audio(input_audio)
        
def predictfile(audio_path):
    # Gets feature of the audio file
    audios_feat = get_features(audio_path)
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
    # Returns prediction
    if y_pred.round() == 1:
        return f'Real,\nour model is {round((float(y_pred[0])-0.5)*200, 2)}% sure'
    else:
        return f'Deepfake,\nour model is {round((abs(float(y_pred[0])-0.5))*200, 2)}% sure'
    
    # Preloaded Audio files
# Define a list of preloaded audio files
preloaded_audio_files = list(Path('frontend/test/').glob('*.wav'))
# Convert the list of Paths to a list of strings containing only the file name
preloaded_audio_files = [Path(path).name for path in preloaded_audio_files]
# Add a dropdown menu to select preloaded audio
selected_audio_file = st.selectbox('Select a preloaded audio file', preloaded_audio_files)
# Then later when you are predicting, you need to use the full path
full_path_preloaded_audio_files = list(Path('frontend/test/').glob('*.wav'))
if selected_audio_file != 'Choose an audio preset...':
    # Get the full path of the selected file
    selected_audio_file_path = next((path for path in full_path_preloaded_audio_files if path.name == selected_audio_file), None)
    if selected_audio_file_path is not None:
        preloaded_audio_result = predictfile(str(selected_audio_file_path))
else:
    selected_audio_file_path = None

if selected_audio_file_path is not None:
    preloaded_audio_result = predictfile(str(selected_audio_file_path))

if selected_audio_file_path != 'Choose an audio preset...':
    preloaded_audio_result = predictfile(selected_audio_file_path)
    
#     # Voice Recording
# # Records 3 seconds in any case
# audio_bytes = audio_recorder(
#   energy_threshold=(-1.0, 1.0),
#   pause_threshold=3.0,
# )
# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")
# if audio_bytes is not None:
#         voice_recordings_result = predictfile(str(selected_audio_file_path))

 #Input audio if wav or mp3
if input_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(input_audio.getvalue())
        temp_audio_file = f.name
    # if the input file is mp3, convert it to wav
    if input_audio.type == 'audio/mp3':
        audio = AudioSegment.from_mp3(temp_audio_file)
        #save as wav
        temp_audio_file = tempfile.mktemp('.wav')
        audio.export(temp_audio_file, format='wav')
    uploaded_audio_result = predictfile(temp_audio_file)
if selected_audio_file_path != 'Choose an audio preset...':
    preloaded_audio_result = predictfile(selected_audio_file_path)
    
    
# Display the results separately
if uploaded_audio_result is not None:
    st.write('Uploaded audio prediction: ', uploaded_audio_result)
if preloaded_audio_result is not None:
    st.write('Preloaded audio prediction: ', preloaded_audio_result)
if voice_recordings_result is not None:
    st.write('Voice recording prediction: ', voice_recordings_result)
    
    
st.write('----------------------------------------------------------------')
# Header
st.header("Deepfake Hunter - Pytorch Version")
st.write("Under development")