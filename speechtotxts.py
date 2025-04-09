from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import torch
import pandas as pd
import kagglehub
import joblib
import numpy as np
from scipy.io import wavfile as wav
import noisereduce as nr
import numpy as np
import librosa as lb
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import scipy.io.wavfile as wav

from scipy.io.wavfile import write
import sounddevice as sd

processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small.en")
def listen():
    # Recording settings
    samplerate = 16000  # Sample rate in Hz (CD quality)
    duration = 7  # Duration in seconds

    print("Recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save as WAV file
    a=write("recorded_audio.wav", samplerate, audio_data)
    print("Audio saved as recorded_audio.wav")
    import IPython.display as ipd
    ipd.Audio("recorded_audio.wav")
    file = "recorded_audio.wav" 
    return file




def stt(file):
    file = "recorded_audio.wav"
    # Load audio file
    audio_input, sample_rate = librosa.load(file, sr=16000)

    # Process the audio input for the model
    inputs = processor(
        audio_input,
        sampling_rate=sample_rate,
        return_tensors="pt"  # Return PyTorch tensors
    )

    # Perform inference (disable gradient calculation)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=448,  # Maximum number of tokens to generate
            num_beams=5  # Beam search for better results
        )

    # Decode the predictions into text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Print transcription
    print("Transcription:", transcription[0])

# Load model
clf = joblib.load("random_forest.pkl")



# Load the audio file
file_path = "recorded_audio.wav"  
rate, audio_data = wav.read(file_path)

if audio_data.dtype == np.int16:
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

# noise reduction
reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.8)



# Save the cleaned audio file
wav.write("cleaned_audio.wav", rate, (reduced_noise * 32768).astype(np.int16))

file_name = "cleaned_audio.wav"
audio_path = os.path.abspath(file_name)


def extract_features(audio_path):
    """Extract MFCC features from an audio file."""
    data, sr = lb.load(audio_path, sr=None, mono=True, offset=1.0, duration=10)
    mfcc = lb.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=256, n_mels=40)

    # Convert to a fixed-size feature vector (mean pooling)
    feature_vector = np.mean(mfcc, axis=1)
    return feature_vector


X_new = extract_features(audio_path)
X_new = np.array(X_new).reshape(1, -1)
X_new = X_new.reshape(1, -1)  # Reshape for ML models

# For LSTM-based models, reshape to (1, time_steps, features)
X_new_lstm = X_new.reshape(1, 1, X_new.shape[1])



def emotion(file_path ):
    # Load the audio file
    
    rate, audio_data = wav.read(file_path)

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

    # noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.8)

   

    # Save the cleaned audio file
    wav.write("cleaned_audio.wav", rate, (reduced_noise * 32768).astype(np.int16))
   

    file_name = "cleaned_audio.wav"
    audio_path = os.path.abspath(file_name)
    X_new = extract_features(audio_path)
    X_new = np.array(X_new).reshape(1, -1)
    X_new = X_new.reshape(1, -1)  # Reshape for ML models
    
    # For LSTM-based models, reshape to (1, time_steps, features)
    X_new_lstm = X_new.reshape(1, 1, X_new.shape[1])
    emotion_pred = clf.predict(X_new)[0]  # Predict label
    emotion_map = {
        1: "Neutral",
        2: "Calm",
        3: "Happy",
        4: "Angry",#
        5: "Excited",#
        6: "chearful",#
        7: "Disgust",
        8: "Surprised"#
    }
    print(f"Predicted Emotion: {emotion_map[emotion_pred]}")



    
    
