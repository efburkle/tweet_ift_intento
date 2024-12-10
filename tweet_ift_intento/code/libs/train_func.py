import os
import pickle

import pandas as pd
from cleaning import *
from constants import *
from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def read_sentiment_data(data_path, sentiment_names):
    """
    Reads sentiment data from CSV files, concatenates them into a single DataFrame,
    and initializes a 'TELECOM' column with zeros.

    Args:
        data_path (str): The path to the directory containing the sentiment CSV files.
        sentiment_names (list of str): A list of filenames for the sentiment data CSV files.

    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated sentiment data with an additional 'TELECOM' column.
    """
    print("Leyendo datos")
    dfs = list(map(lambda x: pd.read_csv(data_path + x), sentiment_names))
    tweets = pd.concat(dfs, ignore_index=True)
    tweets[TELECOM] = 0
    return tweets


def read_ift_data(data_path):
    """
    Reads IFT data from the specified directory path and returns a concatenated DataFrame.

    This function traverses the directory tree rooted at `data_path`, reads all Excel files found,
    and concatenates them into a single DataFrame. The resulting DataFrame contains columns for
    user, tweet, link, and date, with an additional column indicating the telecom status.

    Args:
        data_path (str): The root directory path where the Excel files are located.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all Excel files, with columns
                      ['User', 'Tweet', 'Link', 'Date', 'TELECOM'].
    """
    print("Leyendo datos")
    dfs = list()
    for raiz, directorios, archivos in os.walk(data_path):
        for archivo in archivos:
            dfs.append(
                pd.read_excel(
                    os.path.join(raiz, archivo).replace("\\", "/"), header=None
                )
            )
    tweets = pd.concat(dfs, ignore_index=True)
    tweets.columns = ["User", "Tweet", "Link", "Date"]
    tweets[TELECOM] = 1
    return tweets


def keep_telecom(tweets):
    """
    Filters the given DataFrame to keep only the columns related to tweets and telecom.

    Args:
        tweets (pd.DataFrame): A DataFrame containing tweet data with various columns.

    Returns:
        pd.DataFrame: A DataFrame containing only the columns for tweets and telecom.
    """
    return tweets[[TWEET, TELECOM]]


def equal_sample(tweets1, tweets2, random_state=42):
    def equal_sample(tweets1, tweets2, random_state=42):
        """
        Balances two DataFrames by sampling the larger one to match the size of the smaller one.

        Parameters:
        tweets1 (pd.DataFrame): The first DataFrame containing tweets.
        tweets2 (pd.DataFrame): The second DataFrame containing tweets.
        random_state (int, optional): The seed for the random number generator. Default is 42.

        Returns:
        pd.DataFrame: A concatenated DataFrame containing an equal number of samples from both input DataFrames.
        """

    if len(tweets1) > len(tweets2):
        sample = [tweets1.sample(len(tweets2), random_state=random_state), tweets2]
        return pd.concat(sample, ignore_index=True)
    elif len(tweets1) == len(tweets2):
        sample = [tweets1, tweets2]
        return pd.concat(sample, ignore_index=True)
    else:
        sample = [tweets1, tweets2.sample(len(tweets1), random_state=random_state)]
        return pd.concat(sample, ignore_index=True)


def clean_tweets(tweets, min_len=3, sample=True, frac=0.40, random_state=42):
    """
    Cleans and preprocesses a DataFrame of tweets.
    This function performs the following steps:
    1. Identifies retweets and marks them.
    2. Cleans the tweet text using a custom cleaning function.
    3. Lemmatizes the cleaned tweet text.
    4. Calculates the length of each cleaned tweet.
    5. Optionally samples the DataFrame based on specified criteria.
    Parameters:
    tweets (pd.DataFrame): DataFrame containing tweet data.
    min_len (int, optional): Minimum length of cleaned tweets to keep. Defaults to 3.
    sample (bool, optional): Whether to sample the DataFrame. Defaults to True.
    frac (float, optional): Fraction of the DataFrame to sample if sampling is enabled. Defaults to 0.40.
    random_state (int, optional): Random state for reproducibility when sampling. Defaults to 42.
    Returns:
    pd.DataFrame: Cleaned and optionally sampled DataFrame of tweets.
    """
    print("Limpiando datos")
    tweets[RT] = tweets[TWEET].apply(lambda x: 1 if x.startswith(RT) else 0)
    tweets[CLEANED] = limpieza_total(tweets[TWEET])
    tweets[CLEANED] = tweets[CLEANED].apply(lambda x: lemmatize_text(x))
    tweets[LEN] = tweets[CLEANED].apply(lambda x: len(x.split()))
    if sample:
        tweets = tweets[tweets[RT] == 0][tweets[CLEANED] != ""][
            tweets[LEN] >= min_len
        ].sample(frac=frac, random_state=random_state)
    else:
        tweets = tweets[tweets[RT] == 0][tweets[CLEANED] != ""][tweets[LEN] >= min_len]

    return tweets


def tokenize_tweets(tweets):
    """
    Tokenizes the cleaned tweets by splitting the text into individual words.

    Args:
        tweets (pd.DataFrame): A DataFrame containing the tweets with a column for cleaned text.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column for tokenized tweets.
    """
    print("Tokenizando tweets")
    tweets[TOKEN] = tweets[CLEANED].str.split()
    return tweets


def train_test_split_tweets(tweets, y=SENTIMENT, test_size=0.2, random_state=42):
    """
    Splits the given tweets DataFrame into training and testing sets.

    Parameters:
    tweets (pd.DataFrame): DataFrame containing the tweets data.
    y (str, optional): The column name in the DataFrame representing the target variable. Defaults to SENTIMENT.
    test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

    Returns:
    tuple: A tuple containing four elements:
        - X_train (pd.Series): Training data features.
        - X_test (pd.Series): Testing data features.
        - y_train (pd.Series): Training data target.
        - y_test (pd.Series): Testing data target.
    """
    print("Haciendo split de datos")
    return train_test_split(
        tweets[CLEANED], tweets[y], test_size=test_size, random_state=random_state
    )


def train_vectorizer(
    X_train, min_samples=10, save_vectorizer=True, PATH="../../models/vec_entrenado.pkl"
):
    """
    Trains a TF-IDF vectorizer on the provided training data.

    Parameters:
    X_train (iterable): The training data to fit the vectorizer.
    min_samples (int, optional): The minimum number of documents a term must be present in to be included in the vocabulary. Default is 10.
    save_vectorizer (bool, optional): Whether to save the trained vectorizer to a file. Default is True.
    PATH (str, optional): The file path where the trained vectorizer will be saved if save_vectorizer is True. Default is '../../models/vec_entrenado.pkl'.

    Returns:
    TfidfVectorizer: The trained TF-IDF vectorizer.
    """
    print("Entrenando vectorizador")
    w2v = TfidfVectorizer(min_df=min_samples)
    w2v.fit(X_train)
    if save_vectorizer:
        with open(PATH, "wb") as archivo:
            pickle.dump(w2v, archivo)
    return w2v


def vectorize_tweets(w2v, tweets):
    """
    Vectorizes a list of tweets using a provided word2vec model.

    Args:
        w2v: A word2vec model with a transform method.
        tweets (list of str): A list of tweets to be vectorized.

    Returns:
        list: A list of vectorized tweets.
    """
    print("Vectorizando tweets")
    return w2v.transform(tweets)


def train_model(
    X_train,
    y_train,
    param_grid,
    model,
    cv=5,
    save_model=True,
    PATH="../../models/model_entrenado.pkl",
):
    """
    Trains a machine learning model using GridSearchCV and optionally saves the trained model to a file.

    Parameters:
    X_train (array-like or sparse matrix): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.
    param_grid (dict or list of dictionaries): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
    model (estimator object): This is assumed to implement the scikit-learn estimator interface.
    cv (int, default=5): Determines the cross-validation splitting strategy.
    save_model (bool, default=True): If True, the trained model will be saved to a file specified by PATH.
    PATH (str, default='../../models/model_entrenado.pkl'): The file path where the trained model will be saved if save_model is True.

    Returns:
    estimator object: The estimator that was chosen by the search, i.e. the estimator which gave the highest score (or smallest loss if specified) on the left out data.
    """
    print("Entrenando modelo")
    GS = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    GS.fit(X_train, y_train)
    if save_model:
        with open(PATH, "wb") as archivo:
            pickle.dump(GS.best_estimator_, archivo)
    return GS.best_estimator_


def eval_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the given model using training and test datasets.

    Parameters:
    model : estimator object
        The machine learning model to be evaluated.
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        The training input samples.
    y_train : array-like, shape (n_samples,)
        The target values (class labels) for the training input samples.
    X_test : array-like or sparse matrix, shape (n_samples, n_features)
        The test input samples.
    y_test : array-like, shape (n_samples,)
        The target values (class labels) for the test input samples.

    Returns:
    None

    Prints:
    - The best parameters of the model.
    - The classification report for the test dataset.
    - The classification report for the training dataset.
    """
    print("Evaluando modelo")
    print("Los mejores parametros son: ", model.get_params())
    y_pred = model.predict(X_test)
    print("Reporte de prueba")
    print(classification_report(y_test, y_pred))
    print("Reporte de entrenamiento")
    y_pred = model.predict(X_train)
    print(classification_report(y_train, y_pred))
