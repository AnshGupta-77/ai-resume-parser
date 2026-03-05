import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


def load_and_clean_data(path):

    df = pd.read_csv(path)

    # Combine useful resume fields
    df["resume_text"] = (
        df["career_objective"].astype(str) + " " +
        df["skills"].astype(str) + " " +
        df["major_field_of_studies"].astype(str)
    )

    df["clean_resume"] = df["resume_text"].apply(clean_text)

    return df