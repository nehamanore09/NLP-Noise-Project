import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from scipy.sparse import save_npz, load_npz

print("🔧 FIXED DATA PREP (Sparse Matrix Save)")

# Load original messy tweets
print("1. Loading original 50k messy tweets...")
df = pd.read_csv('tweets_sample.csv')

# NLTK setup
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return 'empty_tweet'
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2 and not t.isdigit()]
    return ' '.join(tokens) if tokens else 'empty_tweet'

# Clean tweets
print("2. Cleaning 50k tweets...")
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'] != 'empty_tweet'].reset_index(drop=True)
print(f"Valid tweets: {len(df)}")

# TF-IDF
print("3. TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# FIXED: Save SPARSE matrices properly
print("4. Saving SPARSE matrices...")
save_npz('X_train.npz', X_train)
save_npz('X_test.npz', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save vectorizer and data
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
df.to_csv('tweets_final.csv', index=False)

print("\n🎉 FIXED FILES SAVED:")
print("- X_train.npz, X_test.npz (SPARSE TF-IDF)")
print("- y_train.npy, y_test.npy")
print("- tfidf_vectorizer.pkl")
print("\n✅ Run 'python train_classifier.py' now!")
