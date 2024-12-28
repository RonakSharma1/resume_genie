from config import EMBEDDINGS_MODEL_NAME,EMBEDDNIGS_FILE_NAME

def get_embedding(client,text):
    response = client.embeddings.create(
        input = text,  
        model=EMBEDDINGS_MODEL_NAME,
        dimensions=256,
    )
    return response.data[0].embedding

def generate_embeddings_and_save_in_memory(resume_latest_data_df, client):
    resume_latest_data_df['embeddings']=resume_latest_data_df.Resume_str.apply(lambda x: get_embedding(client,x))
    resume_latest_data_df.to_pickle(f"{EMBEDDNIGS_FILE_NAME}.pkl")
    
    return resume_latest_data_df
