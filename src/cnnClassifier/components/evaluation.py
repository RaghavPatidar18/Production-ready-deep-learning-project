import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import mlflow
import mlflow.tensorflow
from urllib.parse import urlparse
import os

# Workaround for TensorFlow/Keras version compatibility with MLflow
# Add __version__ attribute to tensorflow.keras if it doesn't exist
if not hasattr(tf.keras, '__version__'):
    tf.keras.__version__ = tf.__version__

import dagshub
dagshub.init(repo_owner='r.patidar181001.1', repo_name='Production-ready-deep-learning-project', mlflow=True)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data / "test",
            shuffle=False,
            **dataflow_kwargs
        )
        
        # Print class indices for verification
        print(f"Evaluation using class indices: {self.valid_generator.class_indices}")

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Try to log the model with error handling for compatibility issues
            try:
                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.tensorflow.log_model(self.model, "model", registered_model_name="BananaRipenessModel")
                else:
                    mlflow.tensorflow.log_model(self.model, "model")
                print("✅ Model logged to MLflow successfully!")
            except AttributeError as e:
                print(f"Error: Could not log model to MLflow due to version compatibility issue: {e}")
                print("Metrics and parameters were still logged successfully!")

    