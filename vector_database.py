from parameters import NUMBER_OF_RESULTS_TO_RETURN, VECTOR_COLLECTION_NAME
import embeddings
import uuid
from pandas import DataFrame
from typing import Any

def store_document_in_vector_db(chroma_client: Any,resume_latest_data_df: DataFrame):
    collection_of_resumes_db = chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)
    
    # use lambda instead of dataframe iteration
    for index, row in resume_latest_data_df.iterrows():
        collection_of_resumes_db.add(
            embeddings = [row["embeddings"]],
            documents = [row["Resume_str"]],
            ids = [str(uuid.uuid4())],
            metadatas = [{"doc_id":str(row["ID"])}]
        )


def query_vector_db_for_user_question(openai_client: Any, chroma_client: Any, question: str):
    # Fetch the vector db collection
    collection_of_resumes_db = chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)

    questions_embedding = embeddings.get_embedding(openai_client, question)

    # Given the above query, identify the top n matching results using vector search
    results = collection_of_resumes_db.query(
        query_embeddings=[questions_embedding],
        n_results=NUMBER_OF_RESULTS_TO_RETURN
    )

    return _extract_optimal_solution(results)
    

def _extract_optimal_solution(vector_db_search_result: Any):
    # extracting the top matching Resume and its ID
    best_result_document = vector_db_search_result.get('documents')[0][0]
    best_result_doc_id = int(vector_db_search_result.get('metadatas')[0][0].get("doc_id"))

    # extracting the IDs of the rest matching Resume(s)
    list_of_resume_ids = [str(vector_db_search_result.get('metadatas')[0][i].get("doc_id")) for i in range(1,NUMBER_OF_RESULTS_TO_RETURN)]
    string_of_resume_ids = ','.join(list_of_resume_ids)

    return best_result_document + f". The id of this specific resume is {best_result_doc_id}. Also, other matching resumes are {string_of_resume_ids}"
