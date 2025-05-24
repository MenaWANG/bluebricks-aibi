import logging
import pandas as pd
from typing import Optional, Union
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    spark
except NameError:
    spark = SparkSession.builder.getOrCreate()

def write_data_to_bricks_catalog(df: Union[DataFrame, pd.DataFrame], 
                               table_name: str, 
                               schema: Optional[str] = None, 
                               mode: str = 'overwrite', 
                               overwriteSchema: bool = True, 
                               mergeSchema: bool = True):
  '''
  Writing data into Unity Catalog.
  
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
  '''

  if isinstance(df, pd.DataFrame):
    df = spark.createDataFrame(df)
    logger.info('pandas df is changed to spark format before writing into catalog')  

  if schema is None:
    target_table_name = f"{table_name}"
  else:
    target_table_name = f"{schema}.{table_name}"
  
  logger.info(f"Attempting to {mode} table to {target_table_name}")

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