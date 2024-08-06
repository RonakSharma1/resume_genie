from config import NUMBER_OF_RESULTS_TO_RETURN, VECTOR_COLLECTION_NAME
import embeddings


def store_document_in_vector_db(chroma_client,resume_latest_data_df):
    collection_of_resumes_db = chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)
    
    # use lambda instead of dataframe iteration
    for index, row in resume_latest_data_df.iterrows():
        collection_of_resumes_db.add(
            embeddings = [row["embeddings"]],
            documents = [row["Resume_str"]],
            ids = [str(row["index"])]
        )


def query_vector_db_for_user_question(azure_client, chroma_client, question,resume_embeddings_df):
    collection_of_resumes_db = chroma_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)

    questions_embedding = embeddings.get_embedding(azure_client, question)

    results = collection_of_resumes_db.query(
        query_embeddings=[questions_embedding],
        n_results=NUMBER_OF_RESULTS_TO_RETURN
    )

    return _extract_optimal_solution(results,resume_embeddings_df)
    

def _extract_optimal_solution(vector_db_search_result, resume_embeddings_df):
    best_result_document = vector_db_search_result.get('documents')[0][0]

    # extracting the top result's string/document and id based on index
    best_result_index = int(vector_db_search_result.get('ids')[0][0])
    best_result_id = resume_embeddings_df.loc[(resume_embeddings_df.index==best_result_index)]['ID']
    best_result_id = int(best_result_id.iloc[0])

    # extracting the id's of other results to provide to user in their response
    list_of_resume_ids = []
    for i in range(1,NUMBER_OF_RESULTS_TO_RETURN):
        other_result_index = int(vector_db_search_result.get('ids')[0][i])
        other_result_id = resume_embeddings_df.loc[(resume_embeddings_df.index==other_result_index)]['ID']
        list_of_resume_ids.append((other_result_id.iloc[0]))

    string_of_resume_ids = ','.join(map(str, list_of_resume_ids))
    best_result = best_result_document + f". The id of this specific resume is {best_result_id}. Also, other matching resumes are {string_of_resume_ids}"
    
    return best_result
