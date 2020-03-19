import pyspark
sc = pyspark.SparkContext('local[*]')

class DB_Caller:
    """
    Wrapper for database calls. Private methods will be called by higher-level `Utils`. Currently only supports Spark.
    """
    def __init__(self):
        self.context = sc
        pass
