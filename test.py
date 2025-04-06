# test.py
import os
from docx2pdf import convert

def convert_docs_to_pdf(directory):
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".docx"):
            docx_path = os.path.join(directory, file)
            convert(docx_path)
            print(f"Converted {file} to PDF")

if __name__ == "__main__":
    directory = 'Resumes'
    convert_docs_to_pdf(directory)
