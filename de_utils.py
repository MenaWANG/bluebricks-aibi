import logging
import pandas as pd
from typing import Optional, Union, List, Dict
from pyspark.sql import SparkSession, DataFrame
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    spark
except NameError:
    spark = SparkSession.builder.getOrCreate()

def validate_dataframe(df: Union[DataFrame, pd.DataFrame], required_columns: List[str]) -> bool:
    """
    Validate that dataframe contains required columns
    
    Parameters:
    -----------
    df : DataFrame or pandas.DataFrame
        The dataframe to validate
    required_columns : List[str]
        List of required column names
        
    Returns:
    --------
    bool : True if valid, raises exception if not
    """
    if isinstance(df, pd.DataFrame):
        df_columns = df.columns.tolist()
    else:
        df_columns = df.columns
    
    missing_columns = [col for col in required_columns if col not in df_columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Dataframe validation passed. Found all required columns: {required_columns}")
    return True

def write_data_to_bricks_catalog(df: Union[DataFrame, pd.DataFrame], 
                               table_name: str, 
                               schema: Optional[str] = None, 
                               mode: str = 'overwrite', 
                               overwriteSchema: bool = True, 
                               mergeSchema: bool = True,
                               validate_columns: Optional[List[str]] = None):
    """
    Writing data into Unity Catalog with enhanced error handling and validation.
    
    Parameters:
    -----------
    df : DataFrame or pandas.DataFrame
        The dataframe to write to the catalog
    table_name : str
        Name of the table to write to
    schema : str, optional
        Schema to use, if None will use just the table_name without schema prefix
    mode : str, default 'overwrite'
        Write mode ('overwrite', 'append', 'ignore', 'error')
    overwriteSchema : bool, default True
        Whether to overwrite the schema if it exists
    mergeSchema : bool, default True
        Whether to merge the schema with existing schema
    validate_columns : List[str], optional
        List of required columns to validate before writing
    """
    
    # Validate dataframe if required columns are specified
    if validate_columns:
        validate_dataframe(df, validate_columns)
    
    # Convert pandas to spark if needed
    if isinstance(df, pd.DataFrame):
        # Check for null values in key columns
        if config.TEXT_COLUMN in df.columns:
            null_count = df[config.TEXT_COLUMN].isnull().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} null values in {config.TEXT_COLUMN} column")
        
        df = spark.createDataFrame(df)
        logger.info('pandas df is changed to spark format before writing into catalog')  

    if schema is None:
        target_table_name = f"{table_name}"
    else:
        target_table_name = f"{schema}.{table_name}"
    
    logger.info(f"Attempting to {mode} table to {target_table_name}")
    
    # Log dataframe info
    logger.info(f"Dataframe shape: {df.count()} rows, {len(df.columns)} columns")

    try:
        df.write \
            .format("delta") \
            .mode(mode) \
            .option("overwriteSchema", str(overwriteSchema).lower()) \
            .option("mergeSchema", str(mergeSchema).lower()) \
            .saveAsTable(target_table_name)
        logger.info(f"Table {target_table_name} written successfully")
    except Exception as e:
        logger.error(f"Error writing table {target_table_name}: {e}")
        raise e

def read_data_from_bricks_catalog(table_name: str, 
                                schema: Optional[str] = None,
                                limit: Optional[int] = None) -> DataFrame:
    """
    Read data from Unity Catalog
    
    Parameters:
    -----------
    table_name : str
        Name of the table to read from
    schema : str, optional
        Schema to use, if None will use just the table_name without schema prefix
    limit : int, optional
        Limit the number of rows to read
        
    Returns:
    --------
    DataFrame : Spark DataFrame
    """
    if schema is None:
        full_table_name = f"{table_name}"
    else:
        full_table_name = f"{schema}.{table_name}"
    
    logger.info(f"Reading data from table: {full_table_name}")
    
    try:
        if limit:
            df = spark.sql(f"SELECT * FROM {full_table_name} LIMIT {limit}")
        else:
            df = spark.sql(f"SELECT * FROM {full_table_name}")
        
        row_count = df.count()
        logger.info(f"Successfully read {row_count} rows from {full_table_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading from table {full_table_name}: {e}")
        raise e

def process_sentiment_analysis_pipeline(source_table: Optional[str] = None,
                                       target_table: Optional[str] = None,
                                       use_mlflow_model: bool = True,
                                       batch_size: int = 1000) -> bool:
    """
    Complete pipeline for sentiment analysis processing
    
    Parameters:
    -----------
    source_table : str, optional
        Source table name (defaults to config.SOURCE_TABLE)
    target_table : str, optional
        Target table name (defaults to config.TARGET_TABLE)
    use_mlflow_model : bool, default True
        Whether to use MLflow registered model or load locally
    batch_size : int, default 1000
        Batch size for processing
        
    Returns:
    --------
    bool : True if successful
    """
    try:
        from mlflow_utils import SentimentAnalyzer
        
        # Use config defaults if not provided
        source_table = source_table or config.SOURCE_TABLE
        target_table = target_table or config.TARGET_TABLE
        
        logger.info(f"Starting sentiment analysis pipeline: {source_table} -> {target_table}")
        
        # Read source data
        df_spark = read_data_from_bricks_catalog(source_table)
        df = df_spark.toPandas()
        
        # Validate required columns
        validate_dataframe(df, [config.TEXT_COLUMN])
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        if use_mlflow_model:
            analyzer.load_model_from_mlflow()
        else:
            analyzer.load_model_locally()
        
        # Process sentiment analysis in batches
        logger.info(f"Processing {len(df)} records in batches of {batch_size}")
        
        sentiment_results = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_texts = batch_df[config.TEXT_COLUMN].tolist()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            batch_results = analyzer.batch_analyze_sentiment(batch_texts)
            sentiment_results.extend(batch_results)
        
        # Add sentiment scores to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        for col in sentiment_df.columns:
            df[col] = sentiment_df[col]
        
        # Write results
        required_columns = [config.TEXT_COLUMN, 'average_score', 'first_chunk_score', 'last_chunk_score']
        write_data_to_bricks_catalog(
            df, 
            target_table, 
            validate_columns=required_columns
        )
        
        logger.info("Sentiment analysis pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {e}")
        raise e