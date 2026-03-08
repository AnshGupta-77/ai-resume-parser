from preprocess import load_and_clean_data
from model import build_features, train_cluster_model

DATA_PATH = "E:/resume-parser-ai/data/resume_dataset.csv"
    
def main():

    df = load_and_clean_data(DATA_PATH)

    print("Dataset Loaded")

    X, vectorizer = build_features(df["clean_resume"])

    print("Features Created")

    model = train_cluster_model(X)

    print("Model Trained")

    df["cluster"] = model.predict(X)

    print(df[["resume_text","cluster"]].head())

if __name__ == "__main__":
    main()
