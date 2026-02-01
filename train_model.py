import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

print("Loading data...")
df = pd.read_csv('Tweets.csv', encoding='utf-8', engine='python')
df = df[['text','sentiment']].sample(n=5000, random_state=42)  # Use smaller sample
df.dropna(inplace=True)
print(f"Data shape: {df.shape}")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = str(text).lower()
    return text.strip()

df['cleaned_text'] = df['text'].apply(clean_text)
print("Text preprocessing completed")

print("Generating embeddings using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=500, max_df=0.9, min_df=2)
embeddings = vectorizer.fit_transform(df['cleaned_text'].tolist()).toarray()
print("Embeddings generated")

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df['sentiment'],
    test_size=0.2,
    random_state=42,
)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, preds))

custom_tweets = [
    "I love this new phone, amazing battery life!",
    "Worst experience ever, totally disappointed.",
    "The movie was okay, nothing special.",
    "Customer support was very helpful and kind.",
    "Delivery was late but product quality is good."
]

clean_custom = [clean_text(t) for t in custom_tweets]
custom_emb = vectorizer.transform(clean_custom).toarray()
preds_custom = model.predict(custom_emb)

print("\nCustom Tweet Predictions:")
for t, p in zip(custom_tweets, preds_custom):
    print(f"{t} --> {p}")

joblib.dump(model, 'sentiment_model.pkl')
print("\nModel saved as sentiment_model.pkl")
