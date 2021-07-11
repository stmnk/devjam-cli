import os 
from dotenv import load_dotenv   
from elasticsearch import Elasticsearch, helpers


load_dotenv()                    
BONSAI_URL = os.environ.get('BONSAI_URL')
client = Elasticsearch(BONSAI_URL)

def corpus_docs(corpus_path):
    for directory, _, filenames in os.walk(f'src/{corpus_path}'):
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            doc_name = filename.replace('.txt', '')
            file = open(file_path, encoding='utf8')
            file_txt = file.read().replace('\n', ' ')
            file.close()
            yield (doc_name, file_txt)

if __name__ == '__main__':
    corpus = [{'name': doc_name, 'text': doc_text} for doc_name, doc_text in corpus_docs('corpus')]

    # print(corpus)

    try:
        index_name = 'bert'
        client.indices.refresh(index_name)
        bert_count = client.cat.count(index_name, params={"format": "json"})
        docs_count = int(bert_count[0]['count'])

        if  docs_count == 0:
            print ("Populating the corpus ...")
            resp = helpers.bulk(client, 
                corpus,
                index = "bert",
            )
            print ("Elasticsearch SUCCESS:", resp)
        else: 
            print('Elasticsearch `bert` index already populated.')
    except Exception as err:
        print("Elasticsearch ERROR:", err)