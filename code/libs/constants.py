# General Constants
PREV_PATH = "../"
DATA_PATH = "../data"
MODEL_PATH = "../models/"
DATA = "data"
MODELS = "models"
TWEET = "Tweet"
RT = "RT"
TOKEN = "Token"
CLEANED = "Cleaned"
LEN = "Len"
SENTIMENT = "sentiment"
SLASH = "/"

# Sentiment Analysis Constants
sentiment_names = ["/tweets_neg_clean.csv", "/tweets_pos_clean.csv"]
param_svc = {"C": [0.2, 0.5, 0.7]}
SENIMENT_VECTORIZER = "tfidf_sentiment.pkl"
SENTIMENT_MODEL = "sentiment_model.pkl"
# IFT Constants
TRAIN_IFT = "/train_ift"
TELECOM = "telecom"
IFT_VECTORIZER = "tfidf_ift.pkl"
IFT_MODEL = "ift_model.pkl"
param_lr = {"C": [0.1, 1, 10]}
