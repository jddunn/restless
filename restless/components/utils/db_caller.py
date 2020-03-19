import pyspark
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("restless-db")
sc = pyspark.SparkContext.getOrCreate(conf=conf)

class DB_Caller:
    """
    Wrapper for database calls. Private methods will be called by higher-level `Utils`. Currently only supports Spark.
    """
    def __init__(self):
        self.context = sc
        pass
