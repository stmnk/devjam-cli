import time
import numpy as np
import tensorflow_hub as hub
from pymilvus_orm import connections, schema, DataType, Collection, list_collections

from q02_populate_corpus import corpus_docs

host = '127.0.0.1'
port = '19530'
connections.add_connection(default={"host": host, "port": port})

# start milvus DB
# cd src/milvus && docker compose up -d

def milvus_distance():
    connections.connect(alias='default')

    print(f"\nList collections...")
    print(list_collections())

    dim = 512
    default_fields = [
        schema.FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
        schema.FieldSchema(name="score", dtype=DataType.DOUBLE),
        schema.FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    default_schema = schema.CollectionSchema(fields=default_fields, description="bert collection")

    print(f"\nCreate collection...")
    collection = Collection(name="bert_milvus", data=None, schema=default_schema)

    print(f"\nList collections...")
    print(list_collections())

    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    print('\nLoading Univ Sent Encoding from TFHub, this can take some time ...\n', '\n')
    model = hub.load(module_url)
    print ('\n', f"Module {module_url} finished loading\n")

    def embed(text_input):
        return model(text_input)

    corpus = [{'name': doc_name, 'text': doc_text} for doc_name, doc_text in corpus_docs('vectorized_corpus')]
    # print([doc['name'] for doc in corpus])
    docs_texts = [doc['text'] for doc in corpus]
    docs_vectors = list(map(lambda x: np.array(x), embed(docs_texts)))

    cp_len = len(docs_vectors)
    collection.insert([[i for i in range(cp_len)], [float(i) for i in range(cp_len)], docs_vectors])
    default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 512}, "metric_type": "L2"}
    collection.create_index(field_name="float_vector", index_params=default_index)
    # print(f"\nCollection entities: {collection.num_entities}")

    question = 'how many languages does bert understand?'
    question_embedding = embed([question])
    question_vector = np.array(question_embedding[0]).tolist()
    
    collection.load()
    topK = 3
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}
    print(f"\nSearching...")
    start_time = time.time()
    result = collection.search([question_vector], "float_vector", search_params, topK, "count > 0")
    end_time = time.time()
    print("search time = %.4fs\n" % (end_time - start_time))
  
    for hits in result:
        for hit in hits:
            print(hit)
    
    collection.drop()


milvus_distance()

# stop milvus DB
# cd src/milvus && docker compose down                               
