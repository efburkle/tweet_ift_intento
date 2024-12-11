import os
import sys
import kfp

# Agregar el directorio raíz "code" al sys.path
code_path = os.path.abspath("code")
if code_path not in sys.path:
    sys.path.append(code_path)

# Constantes
PIPELINE_NAME = "Tweet-Analysis-Pipeline-v1"
PIPELINE_ROOT = "gs://tweet_ift_intento/pipeline_root"

# Importar módulos necesarios
from libs.train_func import read_bigquery_table  # Usamos la función para cargar datos desde BigQuery
#from train_models.train_ift import train_ift  # Función para entrenar modelo IFT
#from train_models.train_sentiment import train_sentiment  # Función para entrenar modelo de Sentiment
from register import upload_model  # Registrar modelos entrenados


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    # Operación 1: Cargar datos desde BigQuery
    data_op = read_bigquery_table(
        project_id=project_id, dataset_id=bq_dataset, table_id=bq_table
    ).set_display_name("Load data from BigQuery")

    # Operación 2: Entrenar modelo IFT
    ift_model_op = train_ift(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train IFT Model")

    # Operación 3: Entrenar modelo de Sentiment Analysis
    sentiment_model_op = train_sentiment(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train Sentiment Model")

    # Operación 4: Registrar modelo IFT
    upload_model(
        project_id=project_id,
        location=location,
        model=ift_model_op.outputs["output_model"],
    ).set_display_name("Register IFT Model")

    # Operación 5: Registrar modelo de Sentiment Analysis
    upload_model(
        project_id=project_id,
        location=location,
        model=sentiment_model_op.outputs["output_model"],
    ).set_display_name("Register Sentiment Model")


if __name__ == "__main__":
    # Compilar el pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"pipeline.yaml"
    )
