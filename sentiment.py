from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
reviews_train = []
for line in open('/home/hemal/ML/SE_project_sentiment/aclImdb/movie_data/full_train.txt', 'r'):
  reviews_train.append(line.strip())

reviews_test = []
for line in open('/home/hemal/ML/SE_project_sentiment/aclImdb/movie_data/full_test.txt', 'r'):
  reviews_test.append(line.strip())


REPLACE_NO_SPACE = re.compile(
    "(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
  reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
  reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

  return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

'''
english_stop_words = stopwords.words('english')


def remove_stop_words(corpus):
  removed_stop_words = []
  for review in corpus:
    removed_stop_words.append(
        ' '.join([word for word in review.split()
                  if word not in english_stop_words])
    )
  return removed_stop_words


no_stop_words = remove_stop_words(reviews_train_clean)
'''
'''
def get_stemmed_text(corpus):
  from nltk.stem.porter import PorterStemmer
  stemmer = PorterStemmer()
  return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


stemmed_reviews = get_stemmed_text(reviews_train_clean)


def get_lemmatized_text(corpus):
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


lemmatized_reviews = get_lemmatized_text(reviews_train_clean)
'''

"""
for c in [0.001, 0.005, 0.01, 0.05, 0.1]:

  svm = LinearSVC(C=c)
  svm.fit(X_train, y_train)
  print("Accuracy for C=%s: %s"
        % (c, accuracy_score(y_val, svm.predict(X_val))))
"""
# Accuracy for C=0.001: 0.88784
# Accuracy for C=0.005: 0.89456
# Accuracy for C=0.01: 0.89376
# Accuracy for C=0.05: 0.89264
# Accuracy for C=0.1: 0.8928



# Final Accuracy: 0.90064


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]

stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(
    binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size=0.75
)


model = LinearSVC(C=0.01)
model.fit(X, target)

import pickle
filename = 'sentimentmodel.pkl'
from sklearn.externals import joblib
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
print("Final Accuracy: %s"
      % accuracy_score(target, loaded_model.predict(X_test)))
'''
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
'''
'''   
 final_model = LogisticRegression(C=0.05)
 final_model.fit(X, target)

 import pickle
 import pandas as pd

 filename = 'lrmodel.pkl'
 prediction = final_model.predict(X_test)
 p = pd.DataFrame(final_model.predict_proba(X_test))

 print(p)

 print("Final Accuracy: %s" % accuracy_score(target, prediction))
 '''
