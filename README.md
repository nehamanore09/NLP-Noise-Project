\# NLP Sentiment Analysis - 75.5% Accuracy on Real Tweets



Built a tweet sentiment classifier that handles messy real-world data (URLs, emojis, mentions, typos).



**# Results**

| Metric | Value |

|--------|-------|

| Test Accuracy | 75.5% |

| Test Set | 9,943 tweets |

| Total Errors | 2,441 |



!\[Confusion Matrix](confusion\_matrix.png)

!\[Error Types](error\_types\_pie.png)





**#Quick Demo**



powershell

nlp\_noise\_env\\Scripts\\Activate.ps1

python improvements.py

Shows live predictions like:



text

'Great service! Love waiting 3 hours 🙄' → POSITIVE (77%)

'Total garbage, hate this app' → NEGATIVE (87%)

Pipeline

1.6M tweets → clean text → TF-IDF → Logistic Regression → 75.5% accuracy





**#Files**

improvements.py - 30 second demo



sentiment\_model.pkl - trained model (40KB)



tfidf\_vectorizer.pkl - text processing



PNGs - evaluation results





**#What I Learned**

Real-world text preprocessing is tough



Sarcasm detection = 16% error rate





