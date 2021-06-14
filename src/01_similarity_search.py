import os
import json
import requests
import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer

def corpus_docs(corpus):
    for directory, _, filenames in os.walk(corpus):
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            doc_name = filename.replace('.txt', '')
            file = open(file_path, encoding='utf8')
            file_txt = file.read().replace('\n', ' ')
            file.close()
            yield (doc_name, file_txt)

corpus = dict([document for document in corpus_docs('corpus')])

def most_similar(corpus, question, n=1):
    documents_tuples = [document_tuple for document_tuple in corpus.items()]
    documents_texts = [document_text for document_text in corpus.values()]
    documents_texts.append(question)
    query_index = len(documents_texts) - 1
    tfidf_matrix = TfidfVectorizer().fit_transform(documents_texts)
    pairwise_similarity = tfidf_matrix * tfidf_matrix.T
    similarity_matrix = pairwise_similarity.toarray() 
    np.fill_diagonal(similarity_matrix, 0)
    question_similarity_vector = list(enumerate(similarity_matrix[query_index]))
    question_most_similar = sorted(question_similarity_vector, key=lambda x: x[1], reverse=True)
    question_most_relevant = question_most_similar[0:n]
    def extract_score(entry): return format(entry[1], ".3f")
    def extract_text(entry): return corpus[documents_tuples[entry[0]][0]]
    def extract_name(entry): return documents_tuples[entry[0]][0]
    def question_relevance_dict(entry):
        return {
            'score': extract_score(entry), 
            'name': extract_name(entry), 
        }
    question_relevance_list = [question_relevance_dict(entry) for entry in question_most_relevant]
    return question_relevance_list


question = 'how many languages does BERT understand in 2019?'
# {'score': '0.168', 'name': 'medium_bert_2'}
# {'score': '0.142', 'name': 'medium_bert_1'}
# {'score': '0.129', 'name': 'wiki_bert'}
# {'score': '0.081', 'name': 'arXiv_bert'}
# {'score': 0.0007148600998334587, 'start': 436, 'end': 439, 'answer': '104'}
# {'score': 0.004559166729450226, 'start': 360, 'end': 363, 'answer': '104'}
# {'score': 0.059281229972839355, 'start': 2456, 'end': 2463, 'answer': 'over 70'}
# {'score': 0.5450839996337891, 'start': 229, 'end': 232, 'answer': '104'}

question = 'how many languages does BERT understand in 2020?'
# {'score': '0.160', 'name': 'medium_bert_2'}
# {'score': '0.124', 'name': 'wiki_bert'}
# {'score': '0.110', 'name': 'medium_bert_1'}
# {'score': '0.065', 'name': 'arXiv_bert'}
# {'score': 0.000150359672261402, 'start': 436, 'end': 439, 'answer': '104'}
# {'score': 0.0006389703485183418, 'start': 2461, 'end': 2463, 'answer': '70'}
# {'score': 0.0000224238228838657, 'start': 360, 'end': 363, 'answer': '104'}
# {'score': 0.006054538767784834, 'start': 229, 'end': 232, 'answer': '104'}

relevant_docs = most_similar(corpus, question, 2)

for doc in relevant_docs:
    print(doc)

relevant_docs_names = [doc['name'] for doc in relevant_docs]
relevant_docs_texts = [corpus[name] for name in relevant_docs_names]

def get_answer(question, passage):
    inputs = {'question': question, 'context': passage}
    payload = json.dumps(inputs)
    QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    method = 'POST'
    response = requests.request(method, QA_URL, data=payload)
    answer = json.loads(response.content.decode("utf-8"))
    return answer

answers_list = [get_answer(question, passage) for passage in relevant_docs_texts]

most_relevant_doc = relevant_docs[0]['name']
passage = corpus[most_relevant_doc]

for answer in answers_list:
    print(answer)

