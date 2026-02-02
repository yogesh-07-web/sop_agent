from pypdf import PdfReader

def load_pdfs(pdf_files):
    documents = []

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        documents.append(text)

    return documents
