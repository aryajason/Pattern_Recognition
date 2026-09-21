import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Map labels to 0 and 1
data['target'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocessing setup
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(cleaned)

data['clean_message'] = data['message'].apply(clean_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    data['clean_message'], data['target'], data.index, test_size=0.2, random_state=42, stratify=data['target']
)

# Feature Extraction for TF-IDF and BoW
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

bow_vec = CountVectorizer()
X_train_bow = bow_vec.fit_transform(X_train)
X_test_bow = bow_vec.transform(X_test)

# Define classifiers
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Evaluation function to test models
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds)
    rec = recall_score(y_te, preds)
    f1 = f1_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)
    return acc, prec, rec, f1, cm, preds

# Print TF-IDF Results
print("TF-IDF Results:")
for name, clf in models.items():
    acc, prec, rec, f1, cm, _ = evaluate_model(clf, X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# Print BoW Results
print("\nBoW Results:")
nb_preds = None
for name, clf in models.items():
    acc, prec, rec, f1, cm, preds = evaluate_model(clf, X_train_bow, X_test_bow, y_train, y_test)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    if name == 'Naive Bayes':
        nb_preds = preds

# Error Analysis using Naive Bayes with BoW predictions
errors = []
for i, original_idx in enumerate(idx_test):
    true_label = y_test.iloc[i]
    pred_label = nb_preds[i]
    
    if true_label != pred_label:
        errors.append((original_idx, true_label, pred_label, data.loc[original_idx, 'message']))
        if len(errors) == 3:
            break

print("\nError Analysis:")
for e in errors:
    true_str = "Spam" if e[1] == 1 else "Ham"
    pred_str = "Spam" if e[2] == 1 else "Ham"
    print(f"Index {e[0]} | True: {true_str}, Predicted: {pred_str}")
    print(f"Message: {e[3]}\n")
