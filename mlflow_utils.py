"""
MLflow utilities for sentiment analysis model management and serving
"""

import logging
import textwrap
import mlflow
import mlflow.transformers
import pandas as pd
from typing import Dict, List, Union, Optional
from transformers import pipeline
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    MLflow-enhanced sentiment analyzer for Blue Bricks project
    """
    
    def __init__(self):
        self.model = None
        self.chunk_size = config.DEFAULT_CHUNK_SIZE
        
    def load_model_locally(self):
        """Load the sentiment analysis model locally for initial registration"""
        logger.info(f"Loading sentiment model: {config.SENTIMENT_MODEL_NAME}")
        self.model = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL_NAME,
            framework=config.SENTIMENT_FRAMEWORK
        )
        return self.model
    
    def register_model_with_mlflow(self, model_version_description: str = "DistilBERT sentiment analysis model"):
        """
        Register the sentiment analysis model with MLflow
        """
        # Set experiment
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name="sentiment_model_registration"):
            # Load the model
            if self.model is None:
                self.load_model_locally()
            
            # Log parameters
            mlflow.log_params({
                "model_name": config.SENTIMENT_MODEL_NAME,
                "framework": config.SENTIMENT_FRAMEWORK,
                "chunk_size": self.chunk_size,
                "task": "sentiment-analysis"
            })
            
            # Log the model
            model_info = mlflow.transformers.log_model(
                transformers_model=self.model,
                artifact_path="sentiment_model",
                registered_model_name=config.MLFLOW_MODEL_NAME,
                task="text-classification",
                metadata={"description": model_version_description}
            )
            
            logger.info(f"Model registered successfully. Model URI: {model_info.model_uri}")
            return model_info
    
    def get_latest_model_version(self) -> str:
        """Get the latest version of the registered model"""
        client = mlflow.tracking.MlflowClient()
        try:
            latest_version = client.get_latest_versions(
                config.MLFLOW_MODEL_NAME, 
                stages=["Production", "Staging", "None"]
            )[0].version
            logger.info(f"Latest model version: {latest_version}")
            return latest_version
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return "1"
    
    def load_model_from_mlflow(self, version: Optional[str] = None):
        """Load model from MLflow registry"""
        if version is None:
            version = self.get_latest_model_version()
        
        model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/{version}"
        logger.info(f"Loading model from MLflow: {model_uri}")
        
        try:
            self.model = mlflow.transformers.load_model(model_uri)
            logger.info("Model loaded successfully from MLflow")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            logger.info("Falling back to local model loading")
            return self.load_model_locally()
    
    def analyze_sentiment_chunk(self, text: str) -> Dict:
        """Analyze sentiment for a single text chunk"""
        if self.model is None:
            self.load_model_from_mlflow()
        
        try:
            result = self.model(text)
            return {
                'label': result[0]['label'],
                'score': result[0]['score']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def summarize_sentiment(self, text: str, chunk_size: Optional[int] = None) -> Dict:
        """
        Analyze sentiment for longer text by breaking it into chunks
        Returns aggregated sentiment scores including first, last, and average chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # Break the text into chunks
        chunks = textwrap.wrap(text, chunk_size)
        
        # Initialize variables to store scores
        scores = []
        first_chunk_score = None
        last_chunk_score = None
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            result = self.analyze_sentiment_chunk(chunk)
            score = result['score']
            scores.append(score)
            
            # Capture scores for the first and last chunks
            if i == 0:
                first_chunk_score = score
            if i == len(chunks) - 1:
                last_chunk_score = score
        
        # Calculate the average score
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'average_score': average_score,
            'first_chunk_score': first_chunk_score,
            'last_chunk_score': last_chunk_score,
            'num_chunks': len(chunks)
        }
    
    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a batch of texts"""
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            
            result = self.summarize_sentiment(text)
            results.append(result)
        
        logger.info("Batch sentiment analysis completed")
        return results


def create_model_serving_endpoint():
    """
    Create or update a model serving endpoint for the sentiment analysis model.
    Note: This requires Databricks Model Serving to be enabled.
    This function is idempotent: it will check if the endpoint exists and skip creation if so.
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.errors import NotFound

        w = WorkspaceClient()

        # Check if the endpoint already exists
        try:
            w.serving_endpoints.get(name=config.MODEL_SERVING_ENDPOINT_NAME)
            logger.info(f"Endpoint '{config.MODEL_SERVING_ENDPOINT_NAME}' already exists. Skipping creation.")
            # If it exists, you might want to update it to the latest version, but for now we'll just return.
            return w.serving_endpoints.get(name=config.MODEL_SERVING_ENDPOINT_NAME)
        except NotFound:
            logger.info(f"Endpoint '{config.MODEL_SERVING_ENDPOINT_NAME}' not found. A new one will be created.")

        # Get the latest model version
        analyzer = SentimentAnalyzer()
        latest_version = analyzer.get_latest_model_version()
        
        endpoint_config = {
            "name": config.MODEL_SERVING_ENDPOINT_NAME,
            "config": {
                "served_models": [
                    {
                        "name": "sentiment-model",
                        "model_name": config.MLFLOW_MODEL_NAME,
                        "model_version": latest_version,
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                    }
                ],
                "traffic_config": {
                    "routes": [
                        {
                            "served_model_name": "sentiment-model",
                            "traffic_percentage": 100,
                        }
                    ]
                },
            },
        }
        
        logger.info(f"Creating model serving endpoint: {config.MODEL_SERVING_ENDPOINT_NAME}")
        endpoint = w.serving_endpoints.create(**endpoint_config)
        logger.info(f"Endpoint created successfully: {endpoint.name}")
        return endpoint
        
    except Exception as e:
        logger.error(f"Error creating or updating model serving endpoint: {e}")
        logger.info("You can create the endpoint manually through the Databricks UI")
        return None


def get_model_serving_predictions(texts: List[str]) -> List[Dict]:
    """
    Get predictions from the served model endpoint
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()
        
        # Prepare the request
        request_data = {
            "inputs": [{"text": text} for text in texts]
        }
        
        # Make the prediction request
        response = w.serving_endpoints.query(
            name=config.MODEL_SERVING_ENDPOINT_NAME,
            dataframe_records=request_data
        )
        
        logger.info(f"Got predictions for {len(texts)} texts from serving endpoint")
        return response.predictions
        
    except Exception as e:
        logger.error(f"Error getting predictions from serving endpoint: {e}")
        logger.info("Falling back to local model inference")
        
        # Fallback to local model
        analyzer = SentimentAnalyzer()
        results = []
        for text in texts:
            result = analyzer.summarize_sentiment(text)
            results.append(result)
        return results 