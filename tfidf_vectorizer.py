import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# Load cleaned tweets
print("Loading cleaned tweets...")
df = pd.read_csv('tweets_cleaned.csv')
print(f"Loaded {len(df)} cleaned tweets")
print("\nCleaned text examples:")
print(df['clean_text'].head(3).tolist())

print("\n" + "="*60)
print("TF-IDF VECTORIZATION (Text → Numbers)")
print("="*60)

# Create TF-IDF Vectorizer (key settings explained below)
vectorizer = TfidfVectorizer(
    max_features=5000,        # Top 5000 most important words
    ngram_range=(1, 2),       # Unigrams + bigrams ("not good" = negative signal)
    min_df=5,                 # Ignore words in <5 tweets (rare noise)
    lowercase=True
)

# Transform text → TF-IDF matrix
print("Converting text to TF-IDF numbers...")
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

print(f"✅ TF-IDF Matrix shape: {X.shape}")
print(f"• 50k tweets → {X.shape[1]} numerical features")
print(f"• Each tweet = 5000-number vector")

# Show top words (highest TF-IDF scores)
feature_names = vectorizer.get_feature_names_out()
top_words = np.argsort(vectorizer.idf_)[:20]
print("\nTop 20 IMPORTANT words (rare, high TF-IDF):")
for i, word_idx in enumerate(top_words):
    print(f"{i+1:2d}. '{feature_names[word_idx]}' (IDF: {vectorizer.idf_[word_idx]:.2f})")

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train: {X_train.shape[0]} tweets | Test: {X_test.shape[0]} tweets")
print("\nSample TF-IDF values (first tweet, top 5 words):")
sample_tweet = X_train[0].toarray()[0]
top_indices = np.argsort(sample_tweet)[-5:][::-1]
for idx in top_indices:
    if sample_tweet[idx] > 0:
        print(f"  '{feature_names[idx]}': {sample_tweet[idx]:.3f}")

# Save for next step
import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("\n✅ Saved 'tfidf_vectorizer.pkl' for predictions")
print("✅ Ready for classification (Step 5)!")
