from pandas import DataFrame
from transformers import GPT2TokenizerFast
import pandas as pd

MAX_NUMBER_OF_CHARACTERS_PER_DOCUMENT = 1200 # Approximately 300 tokens
TOKENISER = GPT2TokenizerFast.from_pretrained("gpt2") # tokeniser

def divide_document_into_smaller_chunks(resume_raw_data_df: DataFrame):
    
    # calculate tokens per resume
    resume_raw_data_df = _calculate_number_of_tokens_per_document(resume_raw_data_df)
    resume_latest_data_df = pd.DataFrame(columns=["ID", "Resume_str", "number_tokens"])
    
    # initialise
    counter = 0
    new_index=0

    for index, row in resume_raw_data_df.iterrows():
        counter = 0
        number_of_split = (row['number_tokens'])*4 // MAX_NUMBER_OF_CHARACTERS_PER_DOCUMENT # multiplying by 4 as each token is apporx 4 characters    
        for _ in range(number_of_split+2):
            row_value = row['Resume_str'][counter: MAX_NUMBER_OF_CHARACTERS_PER_DOCUMENT+counter]
            row_token =len(TOKENISER.encode(row_value))
            resume_latest_data_df.loc[new_index] = [resume_raw_data_df['ID'][index],row_value,row_token]
            counter=MAX_NUMBER_OF_CHARACTERS_PER_DOCUMENT+counter
            new_index+=1

    resume_latest_data_df["index"] =[i for i in range(1, resume_latest_data_df.shape[0]+1)]
    return resume_latest_data_df


def _calculate_number_of_tokens_per_document(resume_raw_data_df: DataFrame):
    resume_raw_data_df['number_tokens'] = resume_raw_data_df.Resume_str.apply(lambda x: len(TOKENISER.encode(x)))
    return resume_raw_data_df
