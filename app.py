import streamlit as st
from src.app import SessionState
from src.app.load_assets import *
from src.utils import inference
GitHub = "https://github.com/mhannani/ZinVert"
WebApp = "https://zinvert.mhannani.com/"


icon(["fab fa-github fa-2x", "fas fa-globe fa-2x"], [GitHub, WebApp])


st.sidebar.subheader("General Settings")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "1. Pick model you'd like to use",
    ("SEQ__2__SEQ__WITHOUT__ATT",
     "SEQ__2__SEQ__WITH__ATT",
     "SEQ__2__SEQ__WITH__TRANSFORMERS")
)

st.title("Welcome to ZinVert")

# setup the state of the app
session_state = SessionState.get(translate_button=False, show_translated_sentence=False)
st.subheader('~ Input Sentence (In Dutch)')
de_sentence = st.text_input('', help='Dutch input sentence to our seq2seq model.', placeholder='Is that real ?')


translate_button = st.button('Translate')

if translate_button:
    en_sentence = inference(de_sentence)
    st.subheader('~ English sentence')
    output = st.text_input('', value=en_sentence, help='English output sentence from seq2seq model.')
