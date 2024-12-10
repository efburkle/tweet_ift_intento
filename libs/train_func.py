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

X_train, X_test, y_train, y_test = train_test_split(
    tweets["CLEANED"], tweets["TELECOM"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(min_df=10)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC()
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
gs_model = GridSearchCV(model, param_grid, cv=5)
gs_model.fit(X_train_vec, y_train)

print("Mejores parámetros:", gs_model.best_params_)
y_pred_train = gs_model.predict(X_train_vec)
y_pred_test = gs_model.predict(X_test_vec)
print("Reporte de entrenamiento:")
print(classification_report(y_train, y_pred_train))
print("Reporte de prueba:")
print(classification_report(y_test, y_pred_test))
