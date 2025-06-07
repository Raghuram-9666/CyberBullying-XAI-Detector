import re

def clean_text(text):
    # lowercase everything
    text = text.lower()
    
    # remove urls
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # remove special characters and numbers, keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
