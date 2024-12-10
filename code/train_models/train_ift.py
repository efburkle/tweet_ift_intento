# Imports
import os
import sys
import warnings

import pandas as pd
from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Configurar credenciales y cliente BigQuery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "libs/intento-c-enlanube-f5a99ccd5ab3.json"
client = bigquery.Client(project="intento-c-enlanube")

project_id = "intento-c-enlanube"
dataset_id = "datos_entrenamiento"

def read_bigquery_table(project_id, dataset_id, table_id):
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    return client.query(query).to_dataframe()

# Leemos los datos desde BigQuery
df_test = read_bigquery_table(project_id, dataset_id, "test")
df_train = read_bigquery_table(project_id, dataset_id, "train")

# Creamos la etiqueta TELECOM: 1 para train, 0 para test (asumiendo misma lógica del snippet original)
df_test["TELECOM"] = 0
df_train["TELECOM"] = 1

tweets = pd.concat([df_test, df_train], ignore_index=True)

# Limpieza básica (en tu código original reemplaza por la función limpieza_total o similar)
def limpieza_basica(text_series):
    # Aquí colocarías tu función real de limpieza (limpieza_total, etc.)
    # Por ejemplo:
    return text_series.str.lower().str.replace(r'[^a-záéíóúüñ ]', '', regex=True).str.strip()

tweets["CLEANED"] = limpieza_basica(tweets["Tweet"])

def clean_tweets(tweets, min_len=3):
    tweets['CLEANED'] = tweets['CLEANED'].astype(str)
    # Filtramos tweets con pocas palabras
    tweets = tweets[tweets["CLEANED"].str.split().apply(len) >= min_len]
    return tweets

tweets = clean_tweets(tweets)

# Tokenización básica (en caso de ser necesario)
def tokenize(text):
    # Ajusta según tu necesidad. Aquí se asume un tokenizado simple por espacios.
    return text.split()

tweets['TOKENS'] = tweets['CLEANED'].apply(tokenize)

# Dividimos en train y test (internos)
X = tweets['CLEANED']
y = tweets['TELECOM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizamos
vectorizer = TfidfVectorizer(min_df=10)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modelo LogisticRegression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predicciones
y_pred_train = model.predict(X_train_vec)
y_pred_test = model.predict(X_test_vec)

# Reportes
print("Reporte de entrenamiento:")
print(classification_report(y_train, y_pred_train))
print("Reporte de prueba:")
print(classification_report(y_test, y_pred_test))
