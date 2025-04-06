# app.py
import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def categorize_resume(uploaded_file, output_directory):
    reader = PdfReader(uploaded_file)
    page = reader.pages[0]
    text = page.extract_text()
    cleaned_resume = cleanResume(text)

    input_features = word_vector.transform([cleaned_resume])
    prediction_id = model.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")
    
    # Create category folder if it doesn't exist
    category_folder = os.path.join(output_directory, category_name)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    
    target_path = os.path.join(category_folder, uploaded_file.name)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return {'filename': uploaded_file.name, 'category': category_name}

def main():
    st.title("Resume Categorizer Application")
    st.subheader("With Python & Machine Learning")
    st.write("Drop your resumes here, and we’ll figure out their job categories for you!")

if __name__ == "__main__":
    main()
