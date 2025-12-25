import streamlit as st
from gensim.models import Word2Vec
import  numpy as np 
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_model():
    model = Word2Vec.load('cbow.model')
    return model
model = load_model()


def sentence_vector(sentence,model):
    words = sentence.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors,axis=0)

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()


st.set_page_config(page_title="Resume-Job Matching",layout ="centered")
st.title("📄 Resume ↔ Job Matching System")
st.write("CBOW + Semantic Similarity")

resume_text = st.text_area("📄 Paste Resume Text")
job_text = st.text_area("🧾 Paste Job Description")


if st.button("Match Resume"):
    if resume_text and job_text:
        resume_vec = sentence_vector(clean_text(resume_text),model)
        job_vec = sentence_vector(clean_text(job_text),model)
        score = cosine_similarity([resume_vec],[job_vec])[0][0]

    if score >=0.75:
        st.success(f"Strong match {score} ")
    elif score >0.5:
        st.success(f"moderate match {score}")
    else:
        st.success(f"weak match {score} ")

