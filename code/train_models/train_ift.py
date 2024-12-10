# Imports
import os
import sys
import warnings

from sklearn.linear_model import LogisticRegression

# from tweet_ift.code.libs.train_func import *


warnings.filterwarnings("ignore")

# Importar librerías locales
# Todas las funciones usadas en este código se encuentran en train_func.py en la carpeta libs
if os.path.isdir(os.path.abspath(os.path.join("code", "libs"))):
    sys.path.append(os.path.abspath(os.path.join("code", "libs")))
    from train_func import *
else:
    raise ModuleNotFoundError(
        "The 'libs' directory does not exist in the specified path."
    )

# Importamos modelo a usar
# Para ver como se llegó a este resultado, ver nb "clasif_telecom.ipynb"
model = LogisticRegression()
# Leemos los datos
tweets = read_sentiment_data(DATA, sentiment_names)
tweets_ift = read_ift_data(DATA + TRAIN_IFT)
# Limpiamos los datos (código de limpieza en la carpeta libs/train_func.py)
tweets = clean_tweets(tweets)
tweets_ift = clean_tweets(tweets_ift, sample=False)
# Igualamos muestras y las juntamos
tweets = equal_sample(tweets, tweets_ift)
# Tokenizamos los tweets
tweets = tokenize_tweets(tweets)
# Dividimos datos en train y test
X_train, X_test, y_train, y_test = train_test_split_tweets(tweets, y=TELECOM)
# Entrenamos vectorizador, se guarda en la carpeta models
vectorizer = train_vectorizer(X_train, PATH=MODELS + SLASH + IFT_VECTORIZER)
# Vectorizamos tweets
X_train = vectorize_tweets(vectorizer, X_train)
X_test = vectorize_tweets(vectorizer, X_test)
# Entrenamos
model = train_model(X_train, y_train, param_lr, model, PATH=MODELS + SLASH + IFT_MODEL)
# Evaluamos modelo
eval_model(model, X_train, y_train, X_test, y_test)
