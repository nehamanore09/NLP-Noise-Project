import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import load_npz


print("🚀 STEP 5: TRAINING SENTIMENT CLASSIFIER")
print("="*50)

# Load your pre-saved ML-ready data (SPARSE matrices)
print("1. Loading TF-IDF data...")
X_train = load_npz('X_train.npz')
X_test = load_npz('X_test.npz')
y_train = np.load('y_train.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)


print(f"✅ Loaded: Train={X_train.shape}, Test={X_test.shape}")

# 2. Train Logistic Regression (perfect for TF-IDF text)
print("\n2. Training Logistic Regression...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 3. Make predictions
print("3. Predicting on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\n✅ Training complete!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# 4. Detailed metrics
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 5. Confusion Matrix (visualize errors)
cm = confusion_matrix(y_test, y_pred)
print("\n📈 CONFUSION MATRIX:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix: Sentiment Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved 'confusion_matrix.png'")

# 6. Save trained model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n🎉 MODEL SAVED: 'sentiment_model.pkl'")
print("\n✅ STEP 5 COMPLETE! Ready for error analysis (Step 6)")
