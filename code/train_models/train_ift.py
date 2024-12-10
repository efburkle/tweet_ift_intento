# Imports
import os
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

def train_ift(project_id, dataset_id, train_table, test_table):
    """
    Entrena un modelo de clasificación usando datos de BigQuery y LogisticRegression.
    
    Args:
        project_id (str): ID del proyecto de Google Cloud.
        dataset_id (str): ID del dataset en BigQuery.
        train_table (str): Nombre de la tabla de entrenamiento.
        test_table (str): Nombre de la tabla de prueba.

    Returns:
        dict: Contiene el modelo entrenado, el vectorizador y los reportes de entrenamiento y prueba.
    """
    def read_bigquery_table(project_id, dataset_id, table_id):
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
        return client.query(query).to_dataframe()

    # Leer datos desde BigQuery
    df_train = read_bigquery_table(project_id, dataset_id, train_table)
    df_test = read_bigquery_table(project_id, dataset_id, test_table)

    # Crear etiquetas (1 para train, 0 para test)
    df_test["TELECOM"] = 0
    df_train["TELECOM"] = 1

    # Concatenar datasets
    tweets = pd.concat([df_train, df_test], ignore_index=True)

    # Limpieza básica
    def limpieza_basica(text_series):
        return text_series.str.lower().str.replace(r'[^a-záéíóúüñ ]', '', regex=True).str.strip()

    tweets["CLEANED"] = limpieza_basica(tweets["Tweet"])

    def clean_tweets(tweets, min_len=3):
        tweets['CLEANED'] = tweets['CLEANED'].astype(str)
        tweets = tweets[tweets["CLEANED"].str.split().apply(len) >= min_len]
        return tweets

    tweets = clean_tweets(tweets)

    # Dividir datos
    X = tweets["CLEANED"]
    y = tweets["TELECOM"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizar
    vectorizer = TfidfVectorizer(min_df=10)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Entrenar modelo
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluar modelo
    y_pred_train = model.predict(X_train_vec)
    y_pred_test = model.predict(X_test_vec)

    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    # Retornar modelo y reportes
    return {
        "model": model,
        "vectorizer": vectorizer,
        "train_report": report_train,
        "test_report": report_test,
    }
