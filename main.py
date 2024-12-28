from config import RAW_DATA, VECTOR_COLLECTION_NAME

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

'''
If embeddings database doesn't exists, read and chunk data 
and generate and store their embeddings in a vector database
'''
 # REDUCE THIS AWAY TO ANOTHER FUNCTION????? TRY THIS BELOW CONDITION ONCE EVERYTHING ELSE IS WORKING
if not chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME).count() > 0:
    
    # Reading data and Chunking into smaller token size
    resume_raw_data_df = pd.read_csv(RAW_DATA)
    resume_latest_data_df = chunking.divide_document_into_smaller_chunks(resume_raw_data_df=resume_raw_data_df)

    # Generate embeddings
    resume_embeddings_df = embeddings.generate_embeddings_and_save_in_memory(resume_latest_data_df, openai_client)
    
    # Store embeddings in vector database for optimised vector search [If csv exists so should database]
    vector_database.store_document_in_vector_db(chroma_client,resume_embeddings_df)

# Ask question to the LLM
while(1):
    question = input("Enter your question about resumes:")
    '''
    Example:
    Does any candidate already has Aviation Mechanic experience? Please summarise their experience and provide their resume id. Also provide ids of other matching resumes   
    '''
    # trial = " Does any candidate already has Aviation Mechanic experience? Please summarise their experience and provide their resume id. Also provide ids of other matching resumes"
    # Identify the best vector result
    best_match_from_db = vector_database.query_vector_db_for_user_question(openai_client, chroma_client, question)

    # Respond to user query with the best result
    response =  query_llm.answer_question(openai_client, best_match_from_db, question)
    print(response)
