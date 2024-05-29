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

        with st.spinner("Analyzing audio..."):
            transcription_result = whisper_model.transcribe(
                audio_path, language="ru", fp16=False)
            transcription_text = transcription_result["text"]
            st.write("Transcription in Russian:")
            st.write(transcription_text)

        with st.spinner("Classify translation..."):
            st.write("Classified Text:")
            result = sentiment(transcription_text)
            st.write(result)
            if (result[0]['label'] == 'LABEL_2'):
                st.write(f"""This sentence is categorized as
                         a good sentence with a score of
                         {result[0]['score']}""")
            elif (result[0]['label'] == 'LABEL_0'):
                st.write(f"""This sentence is categorized as
                         a bad sentence with a score of
                         {result[0]['score']}""")
            else:
                if (result[0]['score'] >= 0.7):
                    st.write(f"""This sentence is categorized as
                             a neutral sentence with a score of
                             {result[0]['score']}""")
                else:
                    st.write("This sentence is difficult to categorize")
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
