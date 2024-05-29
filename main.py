import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
from transformers import pipeline


@st.cache_resource
def load_whisper_model():
    # Have many variants : tiny, base, small, medium or large
    return whisper.load_model("large",)


@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification",
                    model="sismetanin/rubert-ru-sentiment-rusentiment")


whisper_model = load_whisper_model()
sentiment = load_sentiment_model()

st.title("Russian Speech to Text")
st.write("""Upload an audio file in Russian""")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".wav") as tmp_file:
            audio = AudioSegment.from_file(audio_file)
            audio.export(tmp_file.name, format="wav")
            audio_path = tmp_file.name

        st.audio(audio_path, format="audio/wav")

        with st.spinner("Transcribing audio..."):
            transcription_result = whisper_model.transcribe(
                audio_path, language="ru", fp16=False)
            transcription_text = transcription_result["text"]
            st.write("Transcription in Russian:")
            st.write(transcription_text)

        with st.spinner("Classify translation..."):
            st.write("Classified Text:")
            result = sentiment(transcription_text)
            st.write(result)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
