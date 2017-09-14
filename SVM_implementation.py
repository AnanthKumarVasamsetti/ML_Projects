from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC

def letter_only(str):
    return str.isalpha()

def clean_text(docs):
    Lemmatizer = WordNetLemmatizer()
    all_words = set(names.words())
    cleaned_data = []

    for doc in docs:
        cleaned_data.append(' '.join([Lemmatizer.lemmatize(word.lower()) for word in doc.split() if letter_only(word) and word not in all_words]))

    return cleaned_data

categories = ['comp.graphics','sci.space']
data_train = fetch_20newsgroups(subset = 'train', categories = categories, random_state = 42)
data_test = fetch_20newsgroups(subset = 'test', categories = categories, random_state = 42)

cleaned_train_data = clean_text(data_train.data)
label_train = data_train.target

cleaned_test_data = clean_text(data_test.data)
label_test = data_test.target

tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = 'english', max_features = 800)

term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train_data)
term_docs_test = tfidf_vectorizer.transform(cleaned_test_data)

svm = SVC(kernel = 'linear', C = 1.0, random_state = 42)
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print(accuracy)
