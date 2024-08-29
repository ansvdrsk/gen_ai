import os


class Config:
    OPENSEARCH_HOST = '149.130.211.131'  
    OPENSEARCH_PORT = 9200
    OPENSEARCH_USER = "anna"
    OPENSEARCH_PASSWORD = "Qwer0212@"

   
    REDIS_HOST = '127.0.0.1'
    REDIS_PORT = 6379
    REDIS_PASSWORD = None 
    REDIS_USE_SSL = False 

    OCI_CONFIG_FILE = '~/.oci/config'
    OCI_DEFAULT_PROFILE = 'DEFAULT'
    OCI_CHICAGO_PROFILE = 'CHICAGO'
    OCI_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    OCI_COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaafnqwr4bxz4vhjlvfeecxzxh7ztjn7u6szwevj3uwgjthwnt7nebq"
    OCI_NAMESPACE = "oraseemeail"
    OCI_BUCKET_NAME = "anastasiia"

