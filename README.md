# Resume Genie
Allow HR or any other interested party to query large volumes of resumes submitted for a job advertisement. This is powered by an LLM(GPT 4.0)

# Initial Setup
- Create `config.py` at the root level containing the following credentials;
    - OPENAI_API_KEY
    - API_VERSION
    - GPT_MODEL_NAME 
    - EMBEDDINGS_MODEL_NAME
- Store the resumes to load at the root level as `Resumes.csv` with at the least the following columns `ID` and `Resume_str`
- Run `python3 main.py` from the root directory of this repo
Note: The following resume dataset was used: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
    - When asked to `Do you have new resumes to add?`, enter `N`

# Adding New Resumes
- Assuming you already have the resumes stored in a vector db by following the previous section, add any new resumes as follows;
    - Store them as `New_Resumes.csv` at the root level
    - Run `python3 main.py` and enter `Y` when asked to add new resumes
    - Query the application about the new resume
    - Once added, delete `New_Resumes.csv` and store it somewhere in a central repository


# Inner Workings
This application uses the Retrieval-Augmented Generation (RAG) approach to provide GPT-4.0 with a new dataset that it can utilise when addressing user queries. The process consists of the following four main steps:
- GPT-4.0 has a limited context window. Therefore, once the data is loaded, it is divided into chunks, allowing this information to be passed to the GPT model when answering user queries
- Vector embeddings are calculated for the chunks and stored in a vector database
- When a query is made, an equivalent embedding is generated, and the top-matching result is retrieved using an inbuilt vector search within the database
- GPT-4.0 is then provided with both the user's query and the retrieved semantic knowledge from the database to generate an appropriate response.

# Example

## Question
- Does any candidate already has Aviation Mechanic experience? Please summarise their experience and provide their resume id. Also provide ids of other matching resumes

## Response
- Yes, the candidate associated with resume ID 82738323 has aviation mechanic experience. They have five years of experience working with key aircraft systems, and their duties include performing scheduled and unscheduled maintenance, troubleshooting, major and minor repairs, and post-flight inspections. Additionally, they have logged 4859.3 maintenance hours and 1890.1 supervising hours, indicating a significant level of hands-on experience in the field. 


# Knowledge
- Vector Embeddings: Numerical representation of given text thus allowing to perform mathematical operations such as search etc
- Small vs Large embeddings model: Balance between performance, efficiency, storage and cost. Large embeddings are costly and highly accurate as more dimensions to represent that information
- Serialisation:
    - Store Python objects in memory, they need to be converted into a sequence of bytes that the computer can understand. Eg - JSON, XML, HDF5 and Pickle
    - Allows to preserve the state of complex data types like nested dict etc. Eg- When saved as txt, and read back, you cannot access key-value pairs
    - Pickle over csv, as performant, less space and maintains complex data types unlike in CSV(datetime)
- Vector Search: Behind the scene for vector search, the DB is just perform a mathematical operation to identify the closest vectors in the given vector space. Eg: cosine similarity as follows;
```
def cosine_similarity(embedding1, embedding2):
    dot_product = sum(embedding1[i] * embedding2[i] for i in range(len(embedding1)))
    magnitude1 = sum(x**2 for x in embedding1)**0.5
    magnitude2 = sum(x**2 for x in embedding2)**0.5
    return dot_product / (magnitude1 * magnitude2)
```