# tweet_ift
Análisis de tweets relacionado a telecomunicaciones en México.

Este repositorio contiene 2 carpetas principales:

+ **code**: Carpeta donde se encuentran los archivos .py que contienen las funciones para el uso de los modelos y su entrenamiento (train_models)  
            
+ **libs**: Carpeta que contiene los archivos con las clases y funciones utilizadas por los modelos.

## Entrenamiento de modelos
El entrenamiento se puede llevar a cabo ejecutando dos scripts **train_ift.py** y **train_sentiment.py** en la siguiente ruta */code/train_models* estos entrenan los modelos de clasificación de tweet relacionado con telecomunicaciónes y clasificación de sentimiento respectivamente.

Los entrenamientos en el ambiente de pruebas tardan máximo 5 minutos

## Uso de modelos
Para usar los modelos se necesita importar el script **process.py**, dicho archivo contiene el pipeline final del proceso

## Posibles errores
Los errores más comunes se podrían dar por rutas de archivos o imports, en caso de que pase esto, solo se debe de cambiar el formato de ruta hacia donde se apunta.

## Información de las funciones
En los modulos de preprocess.py, train_func, read.py y process.py se detalla como funciona cada una de las funciones, y que es lo que necesita cada una de ellas para funcionar, la mayoría de ellas solo necesita de los tweets, sin embargo, también se pueden modificar los parametros sin necesidad de modificar directamente la función.

## Scripts vacíos
Se dejan los scripts output.py y app.py por si son necesarios para la implementación en GCP

## Análisis previo
El principal análisis se encuentra en los Notebooks **clasif_telecom.ipynb** y **sentiment.ipynb**, en ambos casos se decidió usar TfIdf como vectorizador, ya que arrojó muchos mejores resultados que word2vec
