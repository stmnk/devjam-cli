import numpy as np
import tensorflow_hub as hub
from dotenv import load_dotenv   

from q06_enterprise_elastic import es_connection

load_dotenv()      

es_client = es_connection('ELASTICSEARCH_LOCAL_ENTERPRISE')

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
print('\nLoading Univ Sent Encoding from TFHub, this can take some time ...\n', '\n')
model = hub.load(module_url)
print ('\n', f"Module {module_url} finished loading\n")

def embed(text_input):
    return model(text_input)

question = 'how many languages does bert understand?'
question_embedding = embed([question])
question_vector = np.array(question_embedding[0]).tolist()
# print(question_vector)
# print(len(question_vector))

script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.question_vector, 'vector') + 1.0",
            "params": {"question_vector": question_vector }
        }
    }
}

response = es_client.search(
    index='bert',
    body={
        "size": 2, 
        "query": script_query, 
        "_source": {
            "includes": [
                "name",
                "text",
                # "vector",
            ]
        }
    }
)

print(response)
