import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    
    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


def load_and_clean_data(path):

    df = pd.read_csv(path)

    df['clean_resume'] = df['Resume'].apply(clean_text)

    return df