import os
import json
import requests


BERT_URL = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles=BERT_(language_model)"
method = 'GET'

if not os.path.isfile('./corpus/wiki_bert.txt'): 
    response = requests.request(method, BERT_URL)
    resp_json = json.loads(response.content.decode("utf-8"))
    wiki_bert = resp_json['query']['pages']['62026514']['extract']
    file = open('./corpus/wiki_bert.txt', 'w+')
    file.write(wiki_bert)
    file.close()

file = open('./corpus/wiki_bert.txt', encoding='utf8')
passage = file.read()
file.close()

# google: how many languages does bert understand?
# https://mapmeld.medium.com/a-whole-world-of-bert-f20d6bd47b2f
# https://arxiv.org/abs/1911.03310

# question = 'how many languages does Bert understand?'
# {'score': 0.040411800146102905, 'start': 390, 'end': 393, 'answer': 'two'}
# The original English-language BERT has two models: 
# (1) the BERTBASE: 12 Encoders with 12 bidirectional self-attention heads, and 
# (2) the BERTLARGE: 24 Encoders with 24 bidirectional self-attention heads.

# question = 'how many languages does bert understand?'
# {'score': 0.06729346513748169, 'start': 2456, 'end': 2463, 'answer': 'over 70'}
# On December 9, 2019, it was reported that BERT had been adopted by Google Search for over 70 languages.

# question = 'how many languages does BERT understand?'
# {'score': 0.15555810928344727, 'start': 2456, 'end': 2463, 'answer': 'over 70'}
# On December 9, 2019, it was reported that BERT had been adopted by Google Search for over 70 languages.

# question = 'how many languages does bert understand in 2019?'
# {'score': 0.03678746894001961, 'start': 2456, 'end': 2463, 'answer': 'over 70'}
question = 'how many languages does BERT understand in 2020?'
# {'score': 0.006360830273479223, 'start': 2461, 'end': 2463, 'answer': '70'}

QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
method = 'POST'

# file = open('corpus/medium_bert_1.txt', encoding='utf8')
# {'score': 0.0001582380646141246, 'start': 360, 'end': 363, 'answer': '104'}

# file = open('corpus/medium_bert_2.txt', encoding='utf8')
# {'score': 0.0018622897332534194, 'start': 436, 'end': 439, 'answer': '104'}

file = open('./corpus/arXiv_bert.txt', encoding='utf8')
# {'score': 0.025512443855404854, 'start': 229, 'end': 232, 'answer': '104'}
passage = file.read()
file.close()

inputs = {'question': question, 'context': passage}

payload = json.dumps(inputs)
response = requests.request(method, QA_URL, data=payload)
answer = json.loads(response.content.decode("utf-8"))

print(answer)
