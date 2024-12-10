import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def nltk_download():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def lemmatize_text(text):
    if not isinstance(text, str):
        return ""  # Devuelve una cadena vacía si no es texto
    tokens = word_tokenize(text)
    return " ".join(tokens)

def stem_text(text):
    """
    Esta función aplica stemming a un texto.

    Recibe como input un string
    """
    stemmer = nltk.stem.PorterStemmer()
    tokens = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in tokens])
