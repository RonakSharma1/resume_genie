from parameters import RAW_DATA, VECTOR_COLLECTION_NAME, NEW_RAW_DATA

# Below 3 lines added for older python version
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# libraries
import pandas as pd

# helper functions
import create_client
import chunking
import embeddings
import vector_database
import query_llm

# OpenAI Client
openai_client = create_client.get_openai_client()

# Vector DB Client
chroma_client = create_client.get_chroma_vector_db_client()


def process_and_store_resumes_in_db(file_name: str):
    '''
    If embeddings database doesn't exists, read and chunk data 
    and generate and store their embeddings in a vector database
    '''
    # Reading data
    resume_raw_data_df = pd.read_csv(file_name)
    
    # Chunking data to meet GPT model's token limit size
    resume_latest_data_df = chunking.divide_document_into_smaller_chunks(resume_raw_data_df=resume_raw_data_df)

    # Generate embeddings
    resume_embeddings_df = embeddings.generate_embeddings_and_save_in_memory(resume_latest_data_df, openai_client)
    
    # Store embeddings in vector database for optimised vector search
    vector_database.store_document_in_vector_db(chroma_client,resume_embeddings_df)


# Store resumes in database if non existent
if not chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME).count() > 0:
    process_and_store_resumes_in_db(file_name=RAW_DATA)
else: 
    # Add new resumes to existing database
    user_mode = input("Do you have any new resumes to add? Y/N: ")
    if user_mode == "Y":
        process_and_store_resumes_in_db(file_name=NEW_RAW_DATA)


# Ask question to the LLM
while(1):
    question = input("Enter your question about resumes: ")
    
    # Identifying the top result using inbuilt vector search
    best_match_from_db = vector_database.query_vector_db_for_user_question(openai_client, chroma_client, question)

    # Respond to user query with the best result
    response =  query_llm.answer_question(openai_client, best_match_from_db, question)
    print(response)
