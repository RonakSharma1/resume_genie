from config import AZURE_EMBEDDINGS_MODEL_NAME,EMBEDDNIGS_FILE_NAME


def get_embedding(client,text):
    response = client.embeddings.create(
        input = text,  
        model=AZURE_EMBEDDINGS_MODEL_NAME,
    )
    return response.data[0].embedding

def generate_and_store_embeddings(resume_latest_data_df, client):
    resume_latest_data_df['embeddings']=resume_latest_data_df.Resume_str.apply(lambda x: get_embedding(client,x))
    resume_latest_data_df.to_csv(EMBEDDNIGS_FILE_NAME)
    
    return resume_latest_data_df