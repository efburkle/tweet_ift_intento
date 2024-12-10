# Imports
import os
import sys
import warnings

from sklearn.svm import SVC

print("Current file location:", os.path.abspath(__file__))  # Ruta del archivo actual
print("Current working directory:", os.getcwd())  # Directorio desde el que se ejecuta el script

warnings.filterwarnings("ignore")

# Importar librerías locales
# Todas las funciones usadas en este código se encuentran en train_func.py en la carpeta libs
if os.path.isdir(os.path.abspath(os.path.join("code", "libs"))):
    sys.path.append(os.path.abspath(os.path.join("code", "libs")))
    from ..libs.train_func import *
else:
    raise ModuleNotFoundError(
        "The 'libs' directory does not exist in the specified path."
    )

# Importamos modelo a usar
# Para ver como se llegó a este resultado, ver nb "sentiment.ipynb"
model = SVC()
# Leemos los datos
tweets = read_sentiment_data(DATA, sentiment_names)
# Limpiamos los datos (código de limpieza en la carpeta libs/train_func.py)
tweets = clean_tweets(tweets)
# Tokenizamos los tweets
tweets = tokenize_tweets(tweets)
# Dividimos datos en train y test
X_train, X_test, y_train, y_test = train_test_split_tweets(tweets)
# Entrenamos vectorizador, se guarda en la carpeta models
vectorizer = train_vectorizer(X_train, PATH=MODELS + SLASH + SENIMENT_VECTORIZER)
# Vectorizamos tweets
X_train = vectorize_tweets(vectorizer, X_train)
X_test = vectorize_tweets(vectorizer, X_test)
# Entrenamos modelo, se guarda en la carpeta models
model = train_model(
    X_train, y_train, param_svc, model, PATH=MODELS + SLASH + SENTIMENT_MODEL
)
# Evaluamos modelo
eval_model(model, X_train, y_train, X_test, y_test)
