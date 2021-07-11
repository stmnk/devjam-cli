import os 
import numpy as np
import tensorflow_hub as hub
from dotenv import load_dotenv   
from elasticsearch import Elasticsearch, helpers

from o2_populate_corpus import corpus_docs

load_dotenv()       

# docker pull docker.elastic.co/elasticsearch/elasticsearch:7.13.3
# docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.13.3          
# export ELASTICSEARCH_LOCAL_ENTERPRISE=0.0.0.0:9200

def es_connection(env_var):
    # SEARCHLY_URL = os.environ.get('SEARCHLY_URL')
    ELASTICSEARCH_LOCAL = os.environ.get(env_var) 
    es_client = Elasticsearch(ELASTICSEARCH_LOCAL) # es_client = Elasticsearch(SEARCHLY_URL)    
    return es_client


es_client = es_connection('ELASTICSEARCH_LOCAL_ENTERPRISE')

def create_qa_index(index_name, index_mapping):
    try:
        if not es_client.indices.exists(index_name):
            es_client.indices.create(
                index=index_name, 
                body=index_mapping,  
                # ignore=[400, 404]
            )
            print(f"Successfully Created Index: {index_name}")
        else:
            print(f"Index {index_name} already exists.")
    except Exception as ex:
        print(str(ex))

if __name__ == '__main__':
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    print('\nLoading Univ Sent Encoding from TFHub, this can take some time ...\n', '\n')
    model = hub.load(module_url)
    print ('\n', f"Module {module_url} finished loading\n")

    def embed(text_input):
        return model(text_input)

    corpus = [{'name': doc_name, 'text': doc_text} for doc_name, doc_text in corpus_docs('vectorized_corpus')]
    # print([doc['name'] for doc in corpus])
    docs_texts = [doc['text'] for doc in corpus]
    docs_vectors = embed(docs_texts)

    vectorized_corpus = [ 
        {**corpus[doc_index], 'vector': np.array(docs_vectors[doc_index]).tolist()} 
        for doc_index in range(len(corpus))
    ]

    # print(vectorized_corpus)

    index_mapping = {
        "mappings": {
            "properties": {
                "name": {
                    "type": "text"
                },
                "text": {
                    "type": "text"
                },
                # available in the enterprise ES version 
                "vector": {
                    "type": "dense_vector", 
                    "dims": 512
                }
            }
        }
    }


    try:
        index_name = 'bert'
        if es_client.indices.exists(index_name):
            es_client.indices.delete(index=index_name)

        create_qa_index(index_name, index_mapping)
        print ("Populating the corpus ...")
        resp = helpers.bulk(es_client, 
            vectorized_corpus,
            index = "bert",
        )
        print ("Elasticsearch SUCCESS:", resp)
    except Exception as err:
        print("Elasticsearch ERROR:", err)
