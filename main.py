import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large")


@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_resource
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    return pipeline("translation_ru_to_en", model=model, tokenizer=tokenizer)


whisper_model = load_whisper_model()
translation = load_translation_model()
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
                audio_path, language="ru")
            transcription_text = transcription_result["text"]
            st.write("Transcription in Russian:")
            st.write(transcription_text)

        with st.spinner("Translating transcription..."):
            translated_text = translation(transcription_text)
            st.write("Translation in English:")
            st.write(translated_text[0]['translation_text'])

        with st.spinner("Classify translation..."):
            result = sentiment(translated_text[0]['translation_text'])
            st.write("Classified Text:")
            st.write(result)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
