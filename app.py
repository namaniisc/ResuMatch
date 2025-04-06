# app.py
import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st
import nltk

# Download necessary NLTK resources (run only once)
nltk.download('stopwords')
nltk.download('punkt')

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def basic_pdf_processing(uploaded_file):
    reader = PdfReader(uploaded_file)
    page = reader.pages[0]
    text = page.extract_text()
    cleaned_text = cleanResume(text)
    return cleaned_text

def main():
    st.title("Resume Categorizer Application")
    st.subheader("With Python & Machine Learning")
    st.write("This is a basic setup for processing resumes.")

if __name__ == "__main__":
    main()
