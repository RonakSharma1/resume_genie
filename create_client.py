from config import AZURE_OPENAI_API_KEY,AZURE_API_ENDPOINT,API_VERSION 
from openai import AzureOpenAI
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

def get_azure_openai_client():
    # Azure OpenAI Client
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY, 
        api_version=API_VERSION,
        azure_endpoint = AZURE_API_ENDPOINT
    )  


def get_chroma_vector_db_client():
    return chromadb.PersistentClient(
        path="vector_database",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
