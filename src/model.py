from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def build_features(resumes):

    vectorizer = TfidfVectorizer(max_features=2000)

    X = vectorizer.fit_transform(resumes)

    return X, vectorizer


def train_cluster_model(X, clusters=10):

    model = KMeans(n_clusters=clusters, random_state=42)

    model.fit(X)

    return model