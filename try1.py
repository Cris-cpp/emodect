import streamlit as st
import numpy as np
import librosa as lb
import joblib
import noisereduce as nr
import soundfile as sf

# Load the pre-trained emotion model
def load_model():
    return joblib.load("random_forest.pkl")

# Extract MFCC features for emotion prediction
def extract_features(audio_path):
    data, sr = lb.load(audio_path, sr=None, mono=True, offset=1.0, duration=10)
    mfcc = lb.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=256, n_mels=40)
    return np.mean(mfcc, axis=1)

# Predict emotion from audio
def emotion(file_path):
    data, sr = lb.load(file_path, sr=16000, mono=True)
    reduced_noise = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.8)
    cleaned_path = "cleaned_audio.wav"
    sf.write(cleaned_path, reduced_noise, sr)

    X_new = extract_features(cleaned_path).reshape(1, -1)
    model = load_model()
    emotion_pred = model.predict(X_new)[0]
    emotion_map = {
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Angry",
        5: "Excited", 6: "Cheerful", 7: "Disgust", 8: "Surprised"
    }
    return emotion_map.get(emotion_pred, "Unknown")

# Streamlit UI
def main():
    st.set_page_config(page_title="Emotion Recognition", layout="centered")
    st.title("🎙️ Emotion Recognition from Speech")
    st.markdown("Upload a `.wav` or `.mp3` file to detect emotion (transcription disabled).")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(file_path, format="audio/wav" if file_path.endswith(".wav") else "audio/mp3")

        with st.spinner("🧠 Predicting emotion..."):
            predicted_emotion = emotion(file_path)
        st.success("✅ Emotion prediction complete")
        st.markdown(f"**Predicted Emotion:** :sparkles: _{predicted_emotion}_")

if __name__ == "__main__":
    main()
