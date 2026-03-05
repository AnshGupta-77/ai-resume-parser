from preprocess import load_and_clean_data
from model import build_features, train_cluster_model
from scorer import score_candidate

DATA_PATH = "E:/resume-parser-ai/data/resume_dataset.csv"

skills_required = [
    "python",
    "machine learning",
    "sql",
    "data analysis",
    "deep learning"
]


def main():

    df = load_and_clean_data(DATA_PATH)

    print("Dataset Loaded")

    X, vectorizer = build_features(df['clean_resume'])

    print("Features Created")

    model = train_cluster_model(X)

    print("Model Trained")

    df['cluster'] = model.predict(X)

    print(df[['Category','cluster']].head())

    example_resume = df['clean_resume'][0]

    score = score_candidate(example_resume, skills_required)

    print("Candidate Score:", score,"/10")


if __name__ == "__main__":
    main()