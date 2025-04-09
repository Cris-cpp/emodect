import streamlit as st
import pickle
import numpy as np
from scipy.io import wavfile as wav
import noisereduce as nr
import librosa as lb
import joblib
import os
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def load_model():
    return joblib.load("random_forest.pkl")

def listen():
    samplerate = 16000
    duration = 7
    st.write("Recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    st.write("Recording finished.")
    write("recorded_audio.wav", samplerate, audio_data)
    return "recorded_audio.wav"

def transcribe(file_path):
    processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small.en")
    audio_input, sample_rate = lb.load(file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=448, num_beams=5)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def extract_features(audio_path):
    data, sr = lb.load(audio_path, sr=None, mono=True, offset=1.0, duration=10)
    mfcc = lb.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=256, n_mels=40)
    return np.mean(mfcc, axis=1)

def emotion(file_path):
    rate, audio_data = wav.read(file_path)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.8)
    wav.write("cleaned_audio.wav", rate, (reduced_noise * 32768).astype(np.int16))
    audio_path = os.path.abspath("cleaned_audio.wav")
    X_new = extract_features(audio_path).reshape(1, -1)
    model = load_model()
    emotion_pred = model.predict(X_new)[0]
    emotion_map = {1: "Neutral", 2: "Calm", 3: "Happy", 4: "Angry", 5: "Excited", 6: "Cheerful", 7: "Disgust", 8: "Surprised"}
    return emotion_map[emotion_pred]

def main():
    st.title("Emotion Recognition App")
    if st.button("Record and Predict Emotion"):
        file_path = listen()
        transcription = transcribe(file_path)
        predicted_emotion = emotion(file_path)
        st.write(f"Transcription: **{transcription}**")
        st.write(f"Predicted Emotion: **{predicted_emotion}**")

if __name__ == "__main__":
    main()