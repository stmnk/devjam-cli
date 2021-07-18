import numpy as np
import tensorflow_hub as hub
from dotenv import load_dotenv   
from elasticsearch import Elasticsearch

load_dotenv()      

es_client = Elasticsearch(
    hosts=[{"host": '0.0.0.0', 'port': 9200}], http_auth=('admin', 'admin'),
    scheme='https', verify_certs=False,
)

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

knn_query = {
    "knn": {
        "vector": {
            "vector": question_vector, 
            "k": 3
        }
    }
}

response = es_client.search(
    index='bert',
    body={
        "size": 2, 
        "query": knn_query, 
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
