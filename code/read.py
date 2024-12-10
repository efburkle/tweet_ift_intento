import pickle

from libs.constants import *


def getSentimentVectorizer(PATH=MODEL_PATH + SENIMENT_VECTORIZER):
    """
    Load and return a sentiment vectorizer from a specified file path.

    Args:
        PATH (str): The file path to the sentiment vectorizer. Defaults to MODEL_PATH + SENIMENT_VECTORIZER.

    Returns:
        object: The loaded sentiment vectorizer.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    with open(PATH, "rb") as archivo:
        vectorizador_cargado = pickle.load(archivo)
    return vectorizador_cargado


def getSentimentModel(PATH=MODEL_PATH + SENTIMENT_MODEL):
    """
    Load a sentiment analysis model from a specified file path.

    Args:
        PATH (str): The file path to the sentiment model. Defaults to a combination of MODEL_PATH and SENTIMENT_MODEL.

    Returns:
        object: The loaded sentiment analysis model.
    """
    with open(PATH, "rb") as archivo:
        modelo_cargado = pickle.load(archivo)
    return modelo_cargado


def getIftVectorizer(PATH=MODEL_PATH + IFT_VECTORIZER):
    """
    Load and return the IFT vectorizer from a specified file path.

    Args:
        PATH (str): The file path to the IFT vectorizer. Defaults to MODEL_PATH + IFT_VECTORIZER.

    Returns:
        object: The loaded IFT vectorizer.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    with open(PATH, "rb") as archivo:
        vectorizador_cargado = pickle.load(archivo)
    return vectorizador_cargado


def getIftModel(PATH=MODEL_PATH + IFT_MODEL):
    """
    Load and return the IFT model from the specified file path.

    Args:
        PATH (str): The file path to the IFT model. Defaults to MODEL_PATH + IFT_MODEL.

    Returns:
        object: The loaded IFT model.
    """
    with open(PATH, "rb") as archivo:
        modelo_cargado = pickle.load(archivo)
    return modelo_cargado
