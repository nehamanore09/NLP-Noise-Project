import pandas as pd

# Load 1.6M tweet dataset (takes column names from first row)
print("Loading massive tweet dataset...")
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin-1', 
                 names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

# Show basic info
print(f"Full dataset: {len(df)} rows")
print("\nFirst 3 tweets:")
print(df[['text', 'sentiment']].head(3))

# Map sentiment: 0=negative, 4=positive → 'negative'/'positive'
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

# Take SAMPLE (50k rows) for fast training
df_sample = df[['text', 'sentiment']].dropna().sample(n=50000, random_state=42)
print(f"\nSample dataset: {len(df_sample)} rows")
print("\nSample tweets (messy text!):")
print(df_sample.head())

# Save sample for later
df_sample.to_csv('tweets_sample.csv', index=False)
print("\n✅ Saved 'tweets_sample.csv' (50k messy tweets ready!)")
