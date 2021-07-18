wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip
# other alternatives, tradeof: model size, storage requirements during deployment
# wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
mkdir uncased_L-4_H-512_A-8
unzip uncased_L-4_H-512_A-8.zip -d uncased_L-4_H-512_A-8/
mkdir model && mv uncased_L-4_H-512_A-8/ model && rm uncased_L-4_H-512_A-8.zip