import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import os
# Streamlit page config
st.set_page_config(page_title="Sanskrit–Hindi Translation", layout="wide")
st.title("Sanskrit Translation with Finetuned NLLB Model")

st.markdown("""
# Sanskrit–Hindi Translation App

Welcome! This app uses a fine-tuned **NLLB-200-Distilled-600M** model to translate Sanskrit sentences into Hindi.  
The model has been specifically trained on a domain-specific parallel corpus and achieved a BLEU score of **11.33** and ChrF++ of **37.68** on the test set.

---

### How to Use

- Enter a Sanskrit sentence in the input box below.
- Or, select from the **provided sample sentences** for best results.

⚠️ **Note:** Since the model is large, translation may take a few seconds. For smoother performance, it's recommended to test with the sample inputs.

---
""")

# Model repo and token from secrets.toml
model_name_or_path = "shivrajanand/NLLB-sn-to-hi-finetuned-augmented"
token = st.secrets.get("HF_TOKEN")

st.sidebar.markdown("### Developer Info")
st.sidebar.markdown("""
- **Name:** Shivraj Anand  
- **Email:** shivrajanand022002@gmail.com 
- **GitHub:** [github/shivrajanand](https://github.com/shivrajanand)
- **Portfolio:** [shivrajanand](https://shivrajanand.github.io)
- **Project Details:** 

This is the final year project of B.Tech CSE branch (Batch 2021-25), submitted at B. P. Mandal College of Engineering, Madhepura. 

The work has been carried out under the supervision of Prof. Sujeet Kumar.

The app is experimental and showcases the best model obtained after 226 hours of rigorous training. It aims to demonstrate the practical application of advanced machine learning techniques learned during the course.

For more details visit [BLOG Post](https://shivrajanand.github.io/blog_pages/FinalYearproject/finalyearproject.html) for a complete report.

Note: This is a research-oriented project and may still have room for improvement and optimization.
""")

if not token:
    st.error("Hugging Face token not found. Please set HF_TOKEN in your Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, use_auth_token=token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tokenizer, model, device = load_model_and_tokenizer()

# Load translations CSV (make sure this file is in your repo)
try:
    translations = pd.read_csv("Best_translations.csv")
except FileNotFoundError:
    st.error("Best_translations.csv file not found. Please add it to the app directory.")
    st.stop()

# Cache the sample sentences in session state for better UX
if "sample_df" not in st.session_state:
    st.session_state.sample_df = translations.sample(10).reset_index(drop=True)

sample = st.session_state.sample_df
default_input = sample.iloc[0]['sa'] if not sample.empty else ""

input_sentence = st.text_area("Enter Sanskrit sentence:", value=default_input, height=100)

if st.button("Translate"):
    if not input_sentence.strip():
        st.warning("Please enter a Sanskrit sentence to translate.")
    else:
        with st.spinner("Translating..."):
            try:
                inputs = tokenizer(input_sentence, return_tensors="pt").to(device)
                outputs = model.generate(**inputs)
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.success("Translation complete!")
                st.write("### Translated Output:")
                st.write(translated_text)
            except Exception as e:
                st.error(f"Error during translation: {e}")

st.write("---")
st.write("### Sample Input / Output Sentences for Reference")
st.dataframe(sample[['sa', 'hi_ref']], use_container_width=True)
