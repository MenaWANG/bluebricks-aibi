"""
Configuration settings for Blue Bricks AI/BI Sentiment Analysis Project
"""

# MLflow Configuration
MLFLOW_MODEL_NAME = "bluebricks_sentiment_analyzer"
MLFLOW_EXPERIMENT_NAME = "/Users/shared/bluebricks_sentiment_analysis"

# Model Configuration
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_FRAMEWORK = "pt"
DEFAULT_CHUNK_SIZE = 500

# Data Configuration
SOURCE_TABLE = "bupa_call_synthetic_dataset"
TARGET_TABLE = "bupa_call_synthetic_dataset_with_sentiment_score"
TEXT_COLUMN = "Summary"

# Model Serving Configuration
MODEL_SERVING_ENDPOINT_NAME = "bluebricks-sentiment-endpoint"

# Logging Configuration
LOG_LEVEL = "INFO" 