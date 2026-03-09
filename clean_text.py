def clean_text(text):
    """
    Clean messy tweets but KEEP sarcasm cues (AHHH, cooool)
    """
    if pd.isna(text) or text == '':  # Skip NaN/empty
        return 'empty_tweet'
        
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # 3. Keep basic punctuation, replace others with space
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    
    # 4. Lowercase
    text = text.lower().strip()
    
    # 5. Tokenize + filter
    tokens = word_tokenize(text)
    tokens = [t for t in tokens 
              if t not in stop_words 
              and len(t) > 2 
              and not t.isdigit()]
    
    # Return 'empty_tweet' if no tokens left
    return ' '.join(tokens) if tokens else 'empty_tweet'
