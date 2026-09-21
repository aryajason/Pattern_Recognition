# Pattern Recognition Assignment
**Text Pattern Recognition: From String Matching to Text Classification**

**Author:** Arya Jason Ramadhanto (24/536944/PA/22780)

## Environment Setup
* **Dataset Used:** [SMS Spam Collection Dataset (UCI/Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* **Python Version:** Python 3.14
* **External Libraries:** 
  * `pandas` (Data loading and manipulation)
  * `scikit-learn` (Vectorization and classification models)
  * `nltk` (Text preprocessing, stopwords, PorterStemmer)

## Repository Structure
* `Part_1.py`: Implementation of string matching using regular expressions to extract emails, URLs, phone numbers, monetary amounts, and repeated-character emphasis.
* `Part_2.py`: Preprocessing (lowercasing, punctuation removal, stopword removal, stemming) and construction of statistical text representations (Bag-of-Words, TF-IDF, Bigrams).
* `Part_3.py`: Supervised text classification using Naive Bayes and Logistic Regression, including model evaluation metrics and error analysis logic.
