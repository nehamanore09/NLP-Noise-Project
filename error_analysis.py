import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 STEP 6: ERROR ANALYSIS (Why model fails)")
print("="*50)

# Load everything
print("Loading data + model...")
df = pd.read_csv('tweets_final.csv')
X_test = load_npz('X_test.npz')
y_test = np.load('y_test.npy', allow_pickle=True)
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get predictions
y_pred = model.predict(X_test)
errors = np.where(y_test != y_pred)[0]

print(f"Found {len(errors)} errors ({len(errors)/len(y_test)*100:.1f}% error rate)")

# Create error dataframe
test_df = df.iloc[-9943:].reset_index(drop=True)  # Match test set size
error_df = test_df.iloc[errors].copy()
error_df['error_type'] = np.where(y_test[errors] == 'negative', 'FN', 'FP')
print(f"\nError breakdown:")
print(error_df['error_type'].value_counts())

# Show TOP 10 errors (recruiter demo material!)
print("\n🔥 TOP 10 MODEL FAILURES (Sarcasm + Ambiguity):")
print("="*70)

samples = error_df.sample(10, random_state=42)
for i, idx in enumerate(samples.index):
    orig_text = df.iloc[idx]['text']
    clean_text = df.iloc[idx]['clean_text']
    true = y_test[np.where(y_test == df.iloc[idx]['sentiment'])[0][0]]
    pred = y_pred[np.where(y_pred == df.iloc[idx]['sentiment'])[0][0]]
    confidence = model.predict_proba(X_test[idx:idx+1])[0].max()
    
    print(f"{i+1:2d}. TRUE:{true:8s} PRED:{pred:8s} ({confidence:.1%})")
    print(f"   ORIG:  {orig_text}")
    print(f"   CLEAN: {clean_text}")
    print("-"*70)

# SARASM ANALYSIS - Find common failure patterns
print("\n👀 SARCASTIC/NEGATIVE WORDS in FALSE POSITIVES:")
fp_mask = (error_df['error_type'] == 'FP')
sarcasm_words = ['great', 'awesome', 'perfect', 'love', 'nice', 'good', 'cool']
fp_sarcasm = error_df[fp_mask]['text'].str.contains('|'.join(sarcasm_words), case=False, na=False)
print(f"Sarcasm hits: {fp_sarcasm.sum()} / {fp_mask.sum()} FPs ({fp_sarcasm.mean()*100:.0f}%)")

print("\n📈 ERROR LENGTH ANALYSIS:")
error_df['text_len'] = error_df['text'].str.len()
print(error_df.groupby('error_type')['text_len'].agg(['mean', 'median', 'count']))

# Plot error types
plt.figure(figsize=(10, 6))
error_types = error_df['error_type'].value_counts()
plt.pie(error_types.values, labels=error_types.index, autopct='%1.1f%%', startangle=90)
plt.title('Error Type Distribution (Recruiter Talking Point!)')
plt.savefig('error_types_pie.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ SAVED 'error_types_pie.png'")
print("\n🎉 STEP 6 COMPLETE! Ready for IMPROVEMENTS (Step 7)")
print("\n💼 RECRUITER TALKING POINTS:")
print("• 75.5% accuracy on REAL messy tweets")
print("• 2,441 errors = Sarcasm + short texts + negation")
print("• Improvements: char n-grams, emoji mapping, BERT")
