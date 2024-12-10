import re
import string
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

nltk.download('stopwords')

rid = [
    "las",
    "los",
    "se",
    "les",
    "he",
    "ha",
    "haber",
    "mi",
    "sus",
    "que",
    "el",
    "de",
    "su",
    "lo",
    " ",
    "la",
    "le",
    "e",
    "a",
    "y",
    "me",
    "en",
    "ya",
    "por",
    "si",
    "no",
    "con",
    "un",
    "una",
    "ver",
    "ahi",
    "cómo",
    "entonces",
    "vamos",
    "va",
    "mil",
    "aqui",
    "pues",
    "ahora",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "cien",
    "van",
    "ser",
    "asi",
    "mas",
    "estan",
    "tambien",
    "millones",
    "usted",
    "hacer",
    "ciento",
    "parte",
    "hoy",
    "buenos",
    "dias",
    "senor",
    "gracias",
    "llevar",
    "cabo",
    "presidente",
    "tuitutil",
    "jaja",
    "jajaja",
    "jajajaja",
    "jajajajaja",
    "rt",
    "follow",
    "seguirme",
    "voy",
    "rt",
    "lt",
    "ir",
    "gt",
    "año",
    "ano",
    "dia",
    "mañana",
    "manana",
    "mtvhottest",
    "bien",
    "buen",
]


def clean_text_round0(text):
    """
    Remueve links
    """
    text = unidecode(text.lower())
    text = re.sub(r"http\S+", "", text)
    return text


def clean_text_emoji(text):
    """
    Remueve emojis
    """
    text = unidecode(text.lower())
    text = re.sub(
        r":\)|:\(|:\*|:p|xd|:d|:o|:\||:\-|:\+|:\?|:\!|:\@|:\#|:\$|:\%|:\^|:\&|:\*|:\(|:\)|:\{|\}",
        "",
        text,
    )
    text = emoji.replace_emoji(text, replace="")
    return text


def clean_tweeter_user(text):
    """
    Remueve los usuarios de twitter
    """
    text = re.sub(r"\s?@\w+", "", text)
    return text


def clean_text_round1(text):
    """
    Hace minusculas las palabras, remueve algunos signos de puntuación
    y numeros.

    Esta función limpia a mayor profundidad que clean_text_round1.
    El input debe ser un string.
    """
    text = unidecode(text.lower())
    text = re.sub("\[.*?¿\]\%", " ", text)  # Original -> '\[.*?¿\]\%'
    text = re.sub(
        "[%s]" % re.escape(string.punctuation.replace("_", "") + "¿"), " ", text
    )  # Original -> '[%s]' % re.escape(string.punctuation)
    text = re.sub("\w*\d\w*", " ", text)
    return text


def clean_text_round2(text):
    """
    Se deshace de otros tipos de puntuación que no se limpió en la función pasada.

    Esta función limpia a mayor profundidad que clean_text_round2.

    El input debe ser un string
    """
    text = re.sub("[‘’“”…«»]", "", text)
    text = re.sub("\n", " ", text)
    return text


def clean_text_round3(text):
    """
    Esta función se deshace de los numeros en los
    comentarios.

    Recibe como input un string

    """
    text = re.sub(r"[0-9]", "", text)
    return text


def clean_all(text):
    """
    Aplica las funciones clean_text_round1, clean_text_round2 y
    clean_text_round3 simultaneamente

    Recibe como input un string
    """

    text = clean_text_round0(text)
    text = clean_text_emoji(text)
    text = clean_tweeter_user(text)
    text = clean_text_round1(text)
    text = clean_text_round2(text)
    text = clean_text_round3(text)
    return text


def count_words(serie):
    """
    Cuenta las palabras que existen en una Serie de Pandas

    Input:
        +Serie de Pandas: Esta serie debe de contener LISTAS, no Strings
            las listas se deben de obtener de la siguiente forma

        serie.str.split()

    Output:
        +Serie de Pandas, donde el indice es una palabra, y el valor es el
        conteo de la palabra
    """
    # debe ser una lista con las palabras
    palabras = {}
    serie = serie.reset_index(drop=True)
    for i in range(len(serie)):
        for palabra in serie[i]:
            palabras[palabra] = palabras.get(palabra, 0) + 1
    return palabras


def clean_list(word):
    """
    Se cambian algunas palabras mal escrias o en contracción por palabras originales
    o sin contracción. Si la palabra no está en la lista, regresa la original

    Input:
        +Sring de UNA sola palabra
    """

    diccionario = {
        "tt": "cliente",
        "dev": "devolución",
        "aut": "autorización",
        "mov": "movimiento",
        "msj": "mensaje",
        "zteca": "azteca",
        "atte": "atentamente",
        "cel": "celular",
        "cell": "celular",
        "col": "colonia",
        "eror": "error",
        "erro": "error",
        "fb": "facebook",
        "uds": "ustedes",
        "yono": "yo no",
        "qeu": "que",
        "iso": "hizo",
        "th": "cliente",
        "x": "por",
        "oara": "para",
        "vee": "ver",
        "nk": "no",
        "sw": "",
        "nl": "no",
        "spai": "spei",
        "pp": "app",
        "etsa": "esta",
        "poyo": "apoyo",
        "paea": "para",
        "ub": "un",
        "ze": "",
        "qe": "que",
        "yp": "yo",
        "np": "no",
        "q": "",
        "uma": "una",
        "ttse": "cliente",
        "uba": "una",
        "upf": "ustedes",
        "hivan": "iban",
        "tarbeta": "tarjeta",
        "ununia": "unos",
        "twbgo": "tengo",
        "ahy": "ahi",
        "aparecieon": "aparecieron",
        "ap": "app",
        "bca": "banca",
        "bco": "banco",
        "blq": "bloqueo",
        "aya": "haya",
        "bnco": "banco",
        "bsnco": "banco",
        "c": "",
        "claracion": "aclaracion",
        "cn": "con",
        "ct": "cliente",
        "cta": "cliente",
        "cte": "cliente",
        "ctt": "cliente",
        "d": "",
        "eo": "",
    }
    if word in diccionario.keys():
        word = diccionario[word]
    return word


def correccion_lista(lista, rid=rid, del_stopwords=[]):
    """
    Recibe una lista con palabras individuales y las corrige si es necesario
    una por una.

    Hace uso de la función "clean_list"

    Input:
        +Lista con palabras UNICAS

    Output:
        +Lista con palabras corregidas
    """
    stop_words = set(stopwords.words("spanish") + rid)
    stop_words = [x for x in stop_words if x not in del_stopwords]
    lista_corregida = [
        clean_list(palabra) for palabra in lista if palabra not in stop_words
    ]
    return lista_corregida


def lista_a_frase(lista):
    """
    Pasa una lista de palabras y las unifica en una sola frase.

    Input:
        +Lista con palabras

    Output:
        +String de la lista unificada
    """

    return " ".join(lista)


def limpieza_total(serie, del_stopwords=[]):
    # Si es un string, lo convertimos a una Serie de una sola fila
    if isinstance(serie, str):
        serie = pd.Series([serie])
    else:
        # Caso contrario, asumimos que es una Serie y la convertimos a string
        serie = serie.astype(str)

    serie = (
        serie
        .apply(clean_all)  # Aplica la función para limpieza inicial
        .str.split()       # Separa en palabras
        .apply(lambda x: correccion_lista(x, del_stopwords=del_stopwords))  # Aplica correcciones sobre la lista de palabras
        .apply(lista_a_frase)  # Une las palabras nuevamente en una frase
    )
    return serie
