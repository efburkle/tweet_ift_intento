import os
import pickle
import pandas as pd
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from libs.cleaning import limpieza_total  # Asegúrate de que esta función esté implementada
from libs.constants import TELECOM, TWEET, CLEANED, DATA  # Define estas constantes en 'libs/constants.py'

# Configuración de credenciales para Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "libs/intento-c-enlanube-f5a99ccd5ab3.json"
client = bigquery.Client(project="intento-c-enlanube")

# Funciones para la lectura de datos desde BigQuery
def read_bigquery_table(project_id, dataset_id, table_id):
    """
    Reads a BigQuery table into a pandas DataFrame.
    """
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    return client.query(query).to_dataframe()

def read_sentiment_data(project_id, dataset_id, table_names):
    """
    Reads sentiment data from BigQuery tables and concatenates them into a single DataFrame.
    """
    dfs = []
    for table_name in table_names:
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}`"
        df = client.query(query).to_dataframe()
        dfs.append(df)
    tweets = pd.concat(dfs, ignore_index=True)
    tweets[TELECOM] = 0
    return tweets

def read_ift_data(project_id, dataset_id, table_names):
    """
    Reads IFT data from BigQuery tables and concatenates them into a single DataFrame.
    """
    dfs = []
    for table_name in table_names:
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}`"
        df = client.query(query).to_dataframe()
        dfs.append(df)
    tweets = pd.concat(dfs, ignore_index=True)
    tweets[TELECOM] = 1
    return tweets

# Función para equilibrar los datos
def equal_sample(tweets1, tweets2, random_state=42):
    """
    Balances two DataFrames by sampling the larger one to match the size of the smaller one.
    """
    if len(tweets1) > len(tweets2):
        sample = [tweets1.sample(len(tweets2), random_state=random_state), tweets2]
    elif len(tweets1) == len(tweets2):
        sample = [tweets1, tweets2]
    else:
        sample = [tweets1, tweets2.sample(len(tweets1), random_state=random_state)]
    return pd.concat(sample, ignore_index=True)

# Limpieza de tweets
def clean_tweets(tweets, min_len=3):
    """
    Cleans tweets and ensures a minimum token length.
    """
    if CLEANED not in tweets.columns:
        raise ValueError(f"'{CLEANED}' column is missing in the DataFrame.")
    tweets[CLEANED] = tweets[CLEANED].astype(str)
    tweets = tweets[tweets[CLEANED].str.split().apply(len) >= min_len]
    return tweets

# Vectorización
def train_vectorizer(X_train, min_samples=10, save_vectorizer=True, path="vectorizer.pkl"):
    """
    Trains a TF-IDF vectorizer and saves it to disk.
    """
    print("Entrenando vectorizador")
    vectorizer = TfidfVectorizer(min_df=min_samples)
    vectorizer.fit(X_train)
    if save_vectorizer:
        with open(path, "wb") as f:
            pickle.dump(vectorizer, f)
    return vectorizer

# Entrenamiento del modelo
def train_model(X_train, y_train, param_grid, model, cv=5, save_model=True, path="model.pkl"):
    """
    Trains a machine learning model using GridSearchCV and saves it to disk.
    """
    print("Entrenando modelo")
    gs_model = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    gs_model.fit(X_train, y_train)
    if save_model:
        with open(path, "wb") as f:
            pickle.dump(gs_model.best_estimator_, f)
    return gs_model.best_estimator_

# Evaluación del modelo
def eval_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model and prints classification reports for training and test sets.
    """
    print("Evaluando modelo")
    print("Los mejores parámetros son:", model.get_params())
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print("Reporte de entrenamiento:")
    print(classification_report(y_train, y_pred_train))
    print("Reporte de prueba:")
    print(classification_report(y_test, y_pred_test))

# Flujo principal
if __name__ == "__main__":
    project_id = "intento-c-enlanube"
    dataset_id = "datos_entrenamiento"

    # Leer datos desde BigQuery
    df_test = read_bigquery_table(project_id, dataset_id, "test")
    df_train = read_bigquery_table(project_id, dataset_id, "train")

    # Agregar columna TELECOM
    df_test[TELECOM] = 0
    df_train[TELECOM] = 1

    # Concatenar datos
    tweets = pd.concat([df_test, df_train], ignore_index=True)

    # Limpieza de datos
    print("Limpiando tweets")
    tweets[CLEANED] = limpieza_total(tweets[TWEET])
    tweets = clean_tweets(tweets)

    # Dividir datos en entrenamiento y prueba
    print("Dividiendo datos")
    X_train, X_test, y_train, y_test = train_test_split(
        tweets[CLEANED], tweets[TELECOM], test_size=0.2, random_state=42
    )

    # Vectorización
    print("Vectorizando datos")
    vectorizer = train_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Entrenamiento del modelo
    print("Entrenando modelo")
    model = SVC()
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    best_model = train_model(X_train_vec, y_train, param_grid, model)

    # Evaluación
    eval_model(best_model, X_train_vec, y_train, X_test_vec, y_test)
