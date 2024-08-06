from config import EMBEDDNIGS_FILE_NAME, RAW_DATA

# Below 3 lines added for older python version
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# libraries
import pandas as pd
from ast import literal_eval
import os



# helper functions
import create_client
import chunking
import embeddings
import vector_database
import query_llm




'''
To Do
- Convert to scripts
- Move constants to global variables
- Typing hints???
- UPdate github repo descritpo from chatgpt to gpt4.0
- Should I provide all the top 5 results to llm can extract whatever they want from that info rather than just ids
'''



# Azure OpenAI Client
azure_client = create_client.get_azure_openai_client()

# Vector DB Client
chroma_client = create_client.get_chroma_vector_db_client()


# Checking Embeddings CSV existence
if os.path.isfile(EMBEDDNIGS_FILE_NAME):
    print("world")
    resume_embeddings_df = pd.read_csv(EMBEDDNIGS_FILE_NAME, converters={'embeddings': literal_eval}) # convert string stored embeddings back to list
    print("hello")
else:
    
    ''' CHECK THIS PATH!!!!!!!!!!!! '''
    
    # # Chunking the document
    resume_raw_data_df = pd.read_csv(RAW_DATA)
    resume_latest_data_df = chunking.divide_document_into_smaller_chunks(resume_raw_data_df=resume_raw_data_df)
     
    # generate embeddings
    resume_embeddings_df = embeddings.generate_and_store_embeddings(resume_latest_data_df,azure_client)
    
    # Store resume document in vector database [IF CSV EXISTS THEN SO SHOULD VECTOR DB]
    vector_database.store_document_in_vector_db(chroma_client,resume_embeddings_df)


# Ask question to the LLM
question = input("Enter your question about resumes:")
'''
Example:
Does any candidate already has Aviation Mechanic experience? Please summarise their experience and provide their resume id. Also provide ids of other matching resumes
    
'''
# Identify the best vector result
best_match_from_db = vector_database.query_vector_db_for_user_question(azure_client, chroma_client, question, resume_embeddings_df)


# Respond to user query with the best result
response =  query_llm.answer_question(azure_client, best_match_from_db, question)

print(response)


# EXTRA FUNCTION
# def cosine_similarity(embedding1, embedding2):
#     dot_product = sum(embedding1[i] * embedding2[i] for i in range(len(embedding1)))
#     magnitude1 = sum(x**2 for x in embedding1)**0.5
#     magnitude2 = sum(x**2 for x in embedding2)**0.5
#     return dot_product / (magnitude1 * magnitude2)
  
