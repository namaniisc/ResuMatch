# ResuMatch – Resume Categorization

A Python-based application that automatically categorizes resumes into predefined job roles (e.g., Data Scientist, Java Developer, Business Analyst) using NLP techniques and machine learning models. Includes a Streamlit web interface for easy upload, bulk-processing, and CSV export of results.

---

## 🎬 Demo

![Demo](ResuMatch.gif)

---

## Features

- **Bulk Resume Upload**: Upload multiple PDF resumes at once.
- **Automated Text Extraction**: Uses `PyPDF2` to extract text from the first page of each PDF.
- **Data Cleaning**: Removes URLs, emails, special characters, and stop words via regular expressions and NLTK.
- **Vectorization**: Converts cleaned text into TF-IDF feature vectors.
- **Multi-Class Classification**: Trains and compares several classifiers (KNN, Logistic Regression, Random Forest, SVC, Multinomial NB, OneVsRest).
- **Web Interface**: Streamlit app for file upload, real-time categorization, and CSV download of results.
- **Category-Based Storage**: Organizes processed resumes into folders named after predicted roles.
- **DOC-to-PDF Utility**: Optional script to batch-convert `.doc` files to PDF.

---

## Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - Data Processing & NLP: `pandas`, `NumPy`, `re`, `nltk`
  - Feature Extraction: `scikit-learn` (TF-IDF, LabelEncoder)
  - Machine Learning Models: `scikit-learn` (KNN, LogisticRegression, RandomForestClassifier, SVC, MultinomialNB, OneVsRestClassifier)
  - PDF Parsing: `PyPDF2`
  - Web App: `streamlit`
  - Model Serialization: `pickle`
  - DOC-to-PDF: `docx2pdf` (or equivalent)

- **Environment Management**: `venv` or `conda`

---

## Dataset

- **Source**: Kaggle "Resume Dataset" ([link to dataset])
- **Files**:
  - `updated_resume_dataset.csv` (columns: `category`, `resume`)
  - `resume_dataset.csv` (alternative)
- **Sample Size**: ~960 resumes across ~12 job categories

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/namaniisc/ResuMatch.git
   cd resume-categorizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Jupyter Notebook

1. Launch Jupyter Lab/Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `resume_categorization.ipynb` and run all cells:
   - Data loading & exploration
   - Visualization (bar plots & pie charts)
   - Text cleaning function (`clean_text`)
   - Encoding & TF-IDF vectorization
   - Model training & comparison
   - Save `tfidf_vectorizer.pkl` & `model.pkl`

### Streamlit Web App

1. Ensure `tfidf_vectorizer.pkl` and `model.pkl` are in the root directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. In the browser:
   - Select one or more PDF resumes
   - Specify an output directory (default: `categorized_resumes`)
   - Click **Categorize Resumes**
   - Download the resulting CSV of filenames & predicted categories
   - Check `categorized_resumes/<Category>/` folders for sorted PDFs

---

## File Structure

```
resume-categorizer/
├── app.py                      # Streamlit web application
├── resume_categorization.ipynb # Jupyter notebook
├── model.pkl                   # Trained classification model (Logistic Regression)
├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── utils.py                    # Text cleaning & DOC-to-PDF functions
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

1. **Cleaning**: Removed URLs, emails, special characters.
2. **Tokenization & Stop Word Removal**: Using NLTK’s English stop word list.
3. **Label Encoding**: Converted job categories to numerical labels via `LabelEncoder`.
4. **Vectorization**: Applied `TfidfVectorizer` to convert text into feature vectors.

---

## Model Training & Evaluation

- Split data: 80% train, 20% test (`random_state=42`).
- Baseline classifiers compared:
  - K-Nearest Neighbors
  - Logistic Regression (selected as best)
  - Random Forest
  - Support Vector Classifier
  - Multinomial Naïve Bayes
  - One-vs-Rest Logistic Regression
- **Best Accuracy**: ~99% on test set (Logistic Regression)

---

## Saving & Loading Models

- **Save**:
  ```python
  import pickle

  with open('tfidf_vectorizer.pkl', 'wb') as f:
      pickle.dump(tfidf_vectorizer, f)
  with open('model.pkl', 'wb') as f:
      pickle.dump(model, f)
  ```
- **Load**:
  ```python
  with open('tfidf_vectorizer.pkl', 'rb') as f:
      tfidf_vectorizer = pickle.load(f)
  with open('model.pkl', 'rb') as f:
      model = pickle.load(f)
  ```

---

## DOC-to-PDF Conversion

A utility function (`convert_docs_to_pdf`) uses `docx2pdf` to batch-convert `.doc` files in a directory to PDF:

```python
from docx2pdf import convert
def convert_docs_to_pdf(input_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith('.doc') or filename.endswith('.docx'):
            convert(os.path.join(input_dir, filename))
```

---

## Future Improvements

- Integrate OCR for scanned PDF resumes.
- Add more advanced NLP (Named Entity Recognition).
- Deploy as a REST API (FastAPI / Flask).
- Add user authentication & dashboard.

---

## Acknowledgements

- Kaggle Resume Dataset
- Streamlit Documentation
- Scikit-learn Documentation
- PyPDF2 & NLTK Libraries

