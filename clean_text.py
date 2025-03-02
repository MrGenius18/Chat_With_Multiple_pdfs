import unicodedata
import re

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode
    text = text.encode("utf-8", "ignore").decode("utf-8")  # Remove unsupported chars
    text = re.sub(r'[\ud800-\udfff]', '', text)  # Remove surrogate pairs
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text