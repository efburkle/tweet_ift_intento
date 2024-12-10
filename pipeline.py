import sys
import kfp

sys.path.append("code")

PIPELINE_NAME = "Tweet-Analysis-Pipeline-v1"
PIPELINE_ROOT = "gs://tweet_ift_intento/pipeline_root"


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
#    from tweet_ift_intento.code.libs import train_func
#    from components.evaluation import choose_best_model
#    from components.models import decision_tree, random_forest
    from register import upload_model
    from libs.train_func import read_bigquery_table  # Usamos la función correcta para cargar los datos desde BigQuery
    from train_models import train_ift, train_sentiment  # Funciones para entrenar los modelos de clasificación

    
    if os.path.isdir(os.path.abspath(os.path.join("code", "train_models"))):
        sys.path.append(os.path.abspath(os.path.join("code", "train_models")))
        from train_ift import read_sentiment_data, read_ift_data, clean_tweets, equal_sample, tokenize_tweets, train_test_split_tweets, train_vectorizer, vectorize_tweets, train_model, eval_model
        from train_sentiment import read_sentiment_data, clean_tweets, tokenize_tweets, train_test_split_tweets, train_vectorizer, vectorize_tweets, train_model, eval_model
    else:
        raise ModuleNotFoundError(
            "The 'libs' directory does not exist in the specified path."
        )

    import os
import sys

libs_path = os.path.abspath(os.path.join("code", "libs"))
print(f"Looking for 'libs' directory at: {libs_path}")

if os.path.isdir(libs_path):
    sys.path.append(libs_path)
    print("Directory 'libs' found and added to path.")
else:
    raise ModuleNotFoundError(
        f"The 'libs' directory does not exist in the specified path: {libs_path}"
    )

    
    data_op = read_bigquery_table(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load data from BigQuery")


    ift_model_op = train_ift(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train IFT Model")

    sentiment_model_op = train_sentiment(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train Sentiment Model")
    
    upload_model(
        project_id=project_id,
        location=location,
        model=ift_model_op.outputs["output_model"],
    ).set_display_name("Register IFT Model")

    upload_model(
        project_id=project_id,
        location=location,
        model=sentiment_model_op.outputs["output_model"],
    ).set_display_name("Register Sentiment Model")

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"pipeline.yaml"
    )
