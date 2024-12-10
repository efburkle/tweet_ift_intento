# Imports
import os
import sys
import warnings
import pickle
from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from train_func import *

warnings.filterwarnings("ignore")

# Configuración de credenciales para Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_google_credentials.json"
client = bigquery.Client()

# Definimos constantes
project_id = "intento-c-enlanube"
dataset_id = "datos_entrenamiento"
MODELS = "models"
SLASH = "/"
IFT_VECTORIZER = "ift_vectorizer.pkl"
IFT_MODEL = "ift_model.pkl"
TELECOM = "TELECOM"
CLEANED = "CLEANED"
TWEET = "Tweet"

# Leer los datos desde BigQuery
sentiment_names = ["sentiment_table1", "sentiment_table2"]  # Ajusta según tus tablas
ift_table_names = ["ift_table1", "ift_table2"]  # Ajusta según tus tablas

tweets = read_sentiment_data(project_id, dataset_id, sentiment_names)
tweets_ift = read_ift_data(project_id, dataset_id, ift_table_names)

# Limpiamos los datos
tweets[CLEANED] = limpieza_total(tweets[TWEET])
tweets = clean_tweets(tweets)

tweets_ift[CLEANED] = limpieza_total(tweets_ift[TWEET])
tweets_ift = clean_tweets(tweets_ift, sample=False)

# Igualamos muestras y las juntamos
tweets = equal_sample(tweets, tweets_ift)

# Tokenizamos los tweets
tweets = tokenize_tweets(tweets)

# Dividimos datos en train y test
X_train, X_test, y_train, y_test = train_test_split_tweets(tweets, y=TELECOM)

# Entrenamos vectorizador, se guarda en la carpeta models
vectorizer = train_vectorizer(X_train, PATH=f"{MODELS}{SLASH}{IFT_VECTORIZER}")

# Vectorizamos tweets
X_train = vectorize_tweets(vectorizer, X_train)
X_test = vectorize_tweets(vectorizer, X_test)

# Entrenamos el modelo
param_lr = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
model = train_model(X_train, y_train, param_lr, LogisticRegression(), PATH=f"{MODELS}{SLASH}{IFT_MODEL}")

# Evaluamos modelo
eval_model(model, X_train, y_train, X_test, y_test)
