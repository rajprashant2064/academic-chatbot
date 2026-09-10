import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

def identity_tokenizer(text):
    return text

def identity_preprocessor(text):
    return text

def train_and_save():
    print("Building dataset from intents.json...")

    with open("intents.json", "r") as file:
        data = json.load(file)

    texts = []
    labels = []
    tag_responses = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        patterns = intent["patterns"]
        for pattern in patterns:
            texts.append(pattern.lower())
            labels.append(tag)
        tag_responses[tag] = responses

    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        preprocessor=identity_preprocessor,
        token_pattern=None
    )

    X = vectorizer.fit_transform(texts)

    print("Training classifier (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    print("Computing tag centroids for fuzzy matching...")
    tag_centroids = {}
    for tag in set(labels):
        tag_vectors = X[[i for i, t in enumerate(labels) if t == tag]]
        tag_centroids[tag] = np.mean(tag_vectors.toarray(), axis=0)

    with open("chatbot_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("tag_responses.pkl", "wb") as f:
        pickle.dump(tag_responses, f)

    with open("tag_centroids.pkl", "wb") as f:
        pickle.dump(tag_centroids, f)

    print("Model, vectorizer, and data saved successfully!")

def chatbot_response(user_input, model, vectorizer, tag_responses, tag_centroids):
    user_input = user_input.lower().strip()
    X = vectorizer.transform([user_input])
    predicted_tag = model.predict(X)[0]

    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)

    if confidence < 0.3:
        similarities = {
            tag: cosine_similarity(X.toarray(), centroid.reshape(1, -1))[0][0]
            for tag, centroid in tag_centroids.items()
        }
        predicted_tag = max(similarities, key=similarities.get)

    responses = tag_responses.get(predicted_tag, ["I'm not sure about that yet."])
    return np.random.choice(responses)

if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as e:
        print("Error during training:", str(e))