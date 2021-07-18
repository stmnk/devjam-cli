import os 
from dotenv import load_dotenv   
from elasticsearch import Elasticsearch


load_dotenv()                    
BONSAI_URL = os.environ.get('BONSAI_URL')
client = Elasticsearch(BONSAI_URL)

question = 'how many languages does bert understand?'

query = {
    "more_like_this" : {
        "like" : question,
        "fields" : ["text"],
        "min_term_freq" : 1.9, 
        "min_doc_freq" : 4, 
        "max_query_terms" : 50,
    }
}

result = client.search(index="bert", body={"query": query})

# print(f'Results: {len(result["hits"]["hits"])}')
result_first_two_hits = result['hits']['hits'][:2]
print("First 2 most similar/relevant results: ", '\n')
question_similarity_bonsai = [
    {'name': hit['_source']['name'], 'score': hit['_score'], 'text': hit['_source']['text']}
    for hit in result_first_two_hits
]
for doc in question_similarity_bonsai:
    print(doc)
    print()
