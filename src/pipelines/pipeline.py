import sys

import kfp

sys.path.append("src")

PIPELIE_NAME = "Sentiment_Telecom_pipeline"
PIPELINE_ROOT = "gs://tweet_ift_intento/pipeline_root"


@kfp.dsl.pipeline(name=PIPELIE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
#    from tweet_ift_intento.code.libs import train_func
#    from components.evaluation import choose_best_model
#    from components.models import decision_tree, random_forest
    from components.register import upload_model

    if os.path.isdir(os.path.abspath(os.path.join("tweet_ift_intento","code", "libs"))):
        sys.path.append(os.path.abspath(os.path.join("tweet_ift_intento","code", "libs")))
        from train_func import *
    else:
        raise ModuleNotFoundError(
            "The 'libs' directory does not exist in the specified path."
        )
    if os.path.isdir(os.path.abspath(os.path.join("tweet_ift_intento","code", "train_models"))):
        sys.path.append(os.path.abspath(os.path.join("tweet_ift_intento","code", "train_models")))
        from train_ift import *
        from train_sentiment import *
    else:
        raise ModuleNotFoundError(
            "The 'libs' directory does not exist in the specified path."
        )
    
    data_op = load_data(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load data from BigQuery")

    dt_op = model_vect_ift(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree")

    rf_op = model_vect_sentiments(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest")
    
    upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
    ).set_display_name("Register Model")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"pipeline.yaml"
    )
