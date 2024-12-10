import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def nltk_download():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def lemmatize_text(text):
    """
    Esta función lematiza un texto.

    Recibe como input un string
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])


def stem_text(text):
    """
    Esta función aplica stemming a un texto.

    Recibe como input un string
    """
    stemmer = nltk.stem.PorterStemmer()
    tokens = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in tokens])
