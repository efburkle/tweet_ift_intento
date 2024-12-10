import pandas as pd
from libs.cleaning import *
from libs.constants import *
from libs.preprocess import *
from read import *


def process_sentiment(tweet):
    """
    Processes the sentiment of a given tweet.

    This function performs the following steps:
    1. Retrieves the sentiment vectorizer.
    2. Retrieves the sentiment model.
    3. Cleans the tweet text using the `limpieza_total` function.
    4. Lemmatizes the cleaned tweet text.
    5. Transforms the tweet text into a vector using the sentiment vectorizer.
    6. Predicts the sentiment of the tweet using the sentiment model.

    Args:
        tweet (pd.Series): The tweets to be processed.

    Returns:
        numpy.ndarray: The predicted sentiment of the tweet.
    """
    vec = getSentimentVectorizer()
    model = getSentimentModel()
    tweet = limpieza_total(tweet).apply(lambda x: lemmatize_text(x))
    tweet = vec.transform(tweet)
    return model.predict(tweet)


def process_ift(tweet):
    """
    Processes a tweet using a vectorizer and a model to make a prediction.

    This function performs the following steps:
    1. Retrieves the ift vectorizer.
    2. Retrieves the ift model.
    3. Cleans the tweet text using the `limpieza_total` function.
    4. Lemmatizes the cleaned tweet text.
    5. Transforms the tweet text into a vector using the ift vectorizer.
    6. Predicts the sentiment of the tweet using the ift model.

    Args:
        tweet (pd.Series): The tweets to be processed.

    Returns:
        numpy.ndarray: The prediction result from the model.
    """
    vec = getIftVectorizer()
    model = getIftModel()
    tweet = limpieza_total(tweet).apply(lambda x: lemmatize_text(x))
    tweet = vec.transform(tweet)
    return model.predict(tweet)


def process_tweet(tweet):
    """
    Processes a tweet to analyze its sentiment and IFT (Information Flow Theory).

    Parameters:
    tweet (pd.DataFrame, pd.Series, str, or list): The tweet data to be processed. It can be a DataFrame, Series, string, or list.

    Returns:
    tuple: A tuple containing the results of process_sentiment and process_ift functions applied to the tweet.

    Notes:
    - If the input is a DataFrame, it processes the 'TWEET' column.
    - If the input is a string or list, it converts it to a Series before processing.
    """
    if type(tweet) == pd.DataFrame:
        return process_sentiment(tweet[TWEET]), process_ift(tweet[TWEET])
    elif type(tweet) == str:
        tweet = pd.Series(tweet)
        return process_sentiment(tweet), process_ift(tweet)
    elif type(tweet) == pd.Series:
        return process_sentiment(tweet), process_ift(tweet)
    elif type(tweet) == list:
        tweet = pd.Series(tweet)
        return process_sentiment(tweet), process_ift(tweet)
