import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

print("🔧 COMPLETE DATA PREP + TF-IDF (NaN-proof)")

# Load original messy tweets (source of truth)
print("1. Loading original 50k messy tweets...")
df = pd.read_csv('tweets_sample.csv')

# 2. NLTK setup
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """NaN-proof cleaning"""
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

# 3. Clean ALL tweets
print("2. Cleaning 50k tweets...")
df['clean_text'] = df['text'].apply(clean_text)

# 4. Remove truly empty rows
df = df[df['clean_text'] != 'empty_tweet'].reset_index(drop=True)
print(f"Valid tweets after cleaning: {len(df)}")

# 5. TF-IDF Vectorization
print("3. TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

print(f"✅ TF-IDF Matrix: {X.shape} (rows=valid tweets, cols=features)")

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 7. Save EVERYTHING
import pickle
df.to_csv('tweets_final.csv', index=False)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('X_train.npy', 'wb') as f:
    np.save(f, X_train)
with open('X_test.npy', 'wb') as f:
    np.save(f, X_test)
with open('y_train.npy', 'wb') as f:
    np.save(f, y_train)
with open('y_test.npy', 'wb') as f:
    np.save(f, y_test)

print("\n🎉 STEP 4 COMPLETE! Files saved:")
print("- tweets_final.csv (cleaned data)")
print("- tfidf_vectorizer.pkl (TF-IDF model)")
print("- X_train.npy, y_train.npy (training data)")
print("- X_test.npy, y_test.npy (test data)")
print("\n✅ Ready for Step 5: Classification!")
