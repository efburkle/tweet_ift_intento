import pickle
import pandas as pd
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from .cleaning import *
from .constants import *
from .preprocess import *
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "libs/intento-c-enlanube-f5a99ccd5ab3.json"
client = bigquery.Client(project="intento-c-enlanube")

def read_bigquery_table(project_id, dataset_id, table_id):
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    return client.query(query).to_dataframe()

project_id = "intento-c-enlanube"
dataset_id = "datos_entrenamiento"
df_test = read_bigquery_table(project_id, dataset_id, "test")
df_train = read_bigquery_table(project_id, dataset_id, "train")

df_test["TELECOM"] = 0
df_train["TELECOM"] = 1
tweets = pd.concat([df_test, df_train], ignore_index=True)

# Imprimir las columnas para identificar el nombre correcto
print("Columnas disponibles en tweets:", tweets.columns)

# Suponiendo que la columna con el texto se llama 'text'
# Si el nombre real es otro, cámbialo aquí
tweets["CLEANED"] = limpieza_total(tweets["text"])

def clean_tweets(tweets, min_len=3):
    if 'CLEANED' not in tweets.columns:
        raise ValueError("'CLEANED' column is missing in the DataFrame.")
    tweets['CLEANED'] = tweets['CLEANED'].astype(str)
    tweets = tweets[tweets["CLEANED"].str.split().apply(len) >= min_len]
    return tweets

tweets = clean_tweets(tweets)

