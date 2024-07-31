import streamlit as st
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
st.title("Translation App with Aya 35B")
st.markdown("Translate text from English to various languages using Aya 35B model.")
HUGGING_FACE_TOKEN = st.text_input("Enter your Hugging Face API token:", type="password")
languages = ["French", "German", "Spanish", "Italian", "Hindi", "Portuguese", "Chinese", "Japanese", "Russian", "Korean", "Arabic", "Czech", "Swedish", "Polish", "Romanian", "Vietnamese", "Indonesian", "Turkish", "Greek", "Norwegian", "Bulgarian", "Ukrainian", "Hebrew"]

source_language = st.selectbox("Select source language:", ["English"])
target_language = st.selectbox("Select target language:", languages)
text_to_translate = st.text_area("Enter text to translate:", "")
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "CohereForAI/aya-23-35B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
    model = AutoModel.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
    return tokenizer, model

if HUGGING_FACE_TOKEN:
    tokenizer, model = load_model()
def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer.encode(f"translate {src_lang} to {tgt_lang}: {text}", return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if st.button("Translate"):
    if text_to_translate:
        with st.spinner("Translating..."):
            translated_text = translate_text(text_to_translate, source_language, target_language)
            st.success("Translation completed!")
            st.markdown(f"**Translated Text:** {translated_text}")
    else:
        st.error("Please enter text to translate.")
