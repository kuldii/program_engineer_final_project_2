import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
from transformers import pipeline


@st.cache_resource
def load_whisper_model():
    """
    Loads the 'large' variant of the OpenAI Whisper model. This model is used for
    transcribing audio to text.

    Returns:
        whisper.Whisper: A Whisper model instance loaded with the 'large' variant.
    """
    return whisper.load_model("large")


@st.cache_resource
def load_sentiment_model():
    """
    Loads a sentiment analysis model from Hugging Face's model hub. Specifically, it loads
    the 'sismetanin/rubert-ru-sentiment-rusentiment' model which is fine-tuned for Russian
    language sentiment analysis.

    Returns:
        transformers.Pipeline: A pipeline for text classification with the loaded model.
    """
    return pipeline("text-classification",
                    model="sismetanin/rubert-ru-sentiment-rusentiment")


def interpret_sentiment_result(sent: list) -> str:
    """
    Interprets the sentiment analysis result and returns a formatted string based on the
    label and score.

    Args:
        sent_obj (list): A list containing the sentiment analysis result with 'label' and 'score'.

    Returns:
        str: A formatted string describing the sentiment classification and its confidence.
        :param sent:
    """
    label = sent[0]['label']
    score = sent[0]['score']

    if score > 0.4:
        if label == 'LABEL_2':
            return f"""This sentence is categorized as a good sentence 
                        with a probability of {round(score * 100, 0)}%"""
        elif label == 'LABEL_0':
            return f"""This sentence is categorized as a bad sentence
                        with a probability of {round(score * 100, 0)}%"""
        elif label == 'LABEL_3':
            return f"""This sentence may be bad. Need your attention"""
        else:
            return f"""This sentence is categorized as a neutral sentence 
                        with a probability of {round(score * 100, 0)}%"""
    else:
        return f"""This sentence is difficult to categorize"""


whisper_model = load_whisper_model()
sentiment = load_sentiment_model()

st.title("Classification of Russian speech intonation")
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
            sentiment_text = interpret_sentiment_result(result)
            st.write(sentiment_text)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
