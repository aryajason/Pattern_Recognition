import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# download nltk stopwords
nltk.download('stopwords', quiet=True)

# load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# text preprocessing setup
# added fallback in case your local network blocks NLTK downloads
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

stemmer = PorterStemmer()

def clean_text(text):
    # convert to lowercase and remove punctuation
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove stopwords and apply stemmer
    words = text.split()
    cleaned = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(cleaned)

# apply preprocessing
data['clean_message'] = data['message'].apply(clean_text)

# Bag of Words representation
bow_vec = CountVectorizer()
X_bow = bow_vec.fit_transform(data['clean_message'])
vocab_bow = bow_vec.get_feature_names_out()

# TF-IDF representation
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(data['clean_message'])
vocab_tfidf = tfidf_vec.get_feature_names_out()

# 3. Bigram representation (n=2)
bigram_vec = CountVectorizer(ngram_range=(2, 2))
X_bigram = bigram_vec.fit_transform(data['clean_message'])

# print vocabulary sizes
print(f"BoW Dimensionality: {X_bow.shape[1]}")
print(f"TF-IDF Dimensionality: {X_tfidf.shape[1]}")
print(f"Bigram Dimensionality: {X_bigram.shape[1]}\n")

# Top 10 terms for document index 11
doc_id = 11
bow_arr = X_bow[doc_id].toarray()[0]
tfidf_arr = X_tfidf[doc_id].toarray()[0]

top10_bow_idx = bow_arr.argsort()[-10:][::-1]
top10_tfidf_idx = tfidf_arr.argsort()[-10:][::-1]

print("Top 10 BoW Terms:")
for rank, i in enumerate(top10_bow_idx, 1):
    if bow_arr[i] > 0:
        print(f"{rank}. {vocab_bow[i]}: {bow_arr[i]}")

print("\nTop 10 TF-IDF Terms:")
for rank, i in enumerate(top10_tfidf_idx, 1):
    if tfidf_arr[i] > 0:
        print(f"{rank}. {vocab_tfidf[i]}: {round(tfidf_arr[i], 3)}")

# Cosine similarity calculations
print("\nCosine Similarity:")
sim_bow = cosine_similarity(X_bow[8], X_bow[65])[0][0]
sim_tfidf = cosine_similarity(X_tfidf[8], X_tfidf[65])[0][0]
print(f"Similar Pair (Indices 8 & 65) - BoW: {round(sim_bow, 3)}")
print(f"Similar Pair (Indices 8 & 65) - TF-IDF: {round(sim_tfidf, 3)}")

diff_bow = cosine_similarity(X_bow[8], X_bow[0])[0][0]
diff_tfidf = cosine_similarity(X_tfidf[8], X_tfidf[0])[0][0]
print(f"Dissimilar Pair (Indices 8 & 0) - BoW: {round(diff_bow, 3)}")
print(f"Dissimilar Pair (Indices 8 & 0) - TF-IDF: {round(diff_tfidf, 3)}")
