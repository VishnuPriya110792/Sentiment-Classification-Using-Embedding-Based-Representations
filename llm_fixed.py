import pandas as pd
#importing all the necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('Tweets.csv', encoding='utf-8', engine='python')
# Keep only needed columns
df = df[['text','sentiment']] # Assuming 'text' and 'sentiment' are the relevant columns
# Drop rows with missing values
df.dropna(inplace=True)
print("Data loaded and cleaned")

# Basic EDA
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_dist.png')
plt.close()

# Text length distribution
df['length']=df['text'].astype(str).apply(len)
sns.histplot(data=df, x='length', hue='sentiment', bins=30)
plt.title('Text Length Distribution by Sentiment')
plt.savefig('text_length_dist.png')
plt.close()

# Word Cloud for each sentiment category 
for s in df['sentiment'].unique():
    text = " ".join(df[df['sentiment'] == s]['text'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {s} Sentiment')
    plt.savefig(f'wordcloud_{s}.png')
    plt.close()

print("EDA completed")

# Text Preprocessing 
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = str(text).lower()                   # Convert to lowercase
    return text.strip()
df['cleaned_text'] = df['text'].apply(clean_text)# Clean the text data

print("Text preprocessing completed")

# Embedding and Model Training
try:
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')# Load pre-trained SBERT model
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)# Generate embeddings
    embedding_backend = 'sbert'# If SentenceTransformer is available, use it for embeddings
    print('Using SentenceTransformer for embeddings.')
except Exception as e:
    print('SentenceTransformer unavailable, falling back to TF-IDF:', e) # Fallback to TF-IDF if SBERT is not available
    vectorizer = TfidfVectorizer(max_features=5000)# Initialize TF-IDF Vectorizer
    embeddings = vectorizer.fit_transform(df['cleaned_text'].tolist()).toarray() # Generate TF-IDF embeddings
    embedding_backend = 'tfidf' # If SentenceTransformer is not available, use TF-IDF for embeddings

print("Embeddings generated")

X_train, X_test, y_train, y_test = train_test_split( # Split the data into training and testing sets
    embeddings, df['sentiment'], # Target variable
    test_size=0.2, # 20% for testing
    random_state=42,# For reproducibility
)

print("Training models...")

lr=LogisticRegression(max_iter=1000) # Initialize Logistic Regression model
lr.fit(X_train, y_train) # Train the model
lr_preds = lr.predict(X_test) # Make predictions

le=LabelEncoder() # Initialize Label Encoder
y_train_enc = le.fit_transform(y_train) # Encode target labels
y_test_enc = le.transform(y_test) #

xgb=XGBClassifier() # Initialize XGBoost Classifier
xgb.fit(X_train, y_train_enc) # Train the model

xgb_preds = xgb.predict(X_test) # Make predictions
xgb_preds = le.inverse_transform(xgb_preds) # Decode predictions

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_preds)) #Print classification report for Logistic Regression
print("XGBoost Classification Report:\n")
print(classification_report(y_test, xgb_preds)) #Print classification report for XGBoost

ConfusionMatrixDisplay.from_predictions(y_test, xgb_preds) # Plot confusion matrix
plt.savefig('confusion_matrix.png')
plt.close()

custom_tweets = [ 
    "I love this new phone, amazing battery life!",
    "Worst experience ever, totally disappointed.",
    "The movie was okay, nothing special.",
    "Customer support was very helpful and kind.",
    "Delivery was late but product quality is good."
] # Custom tweets for prediction

clean_custom = [clean_text(t) for t in custom_tweets]

if 'sbert' == globals().get('embedding_backend'): # Check which embedding method was used
    custom_emb = model.encode(clean_custom) # Generate SBERT embeddings for custom tweets
else:
    custom_emb = vectorizer.transform(clean_custom).toarray() 
# Generate TF-IDF embeddings for custom tweets
preds = xgb.predict(custom_emb)
preds = le.inverse_transform(preds)

# Print predictions for custom tweets
print("\nCustom Tweet Predictions:")
for t,p in zip(custom_tweets,preds):
    print(f"{t} --> {p}")

# Save the model for use in the app
import joblib
joblib.dump(xgb, 'sentiment_model.pkl')
print("\nModel saved as sentiment_model.pkl")
