wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

mkdir squad_2
mv dev-v2.0.json train-v2.0.json squad_2

# other alternatives, tradeofs: 
# natural questions: q-a collected from google searches, larger
# coqa: q-a in a conversational context, dialogue

# train_path = keras.utils.get_file("train.json", 
# "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
# "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
# eval_path = keras.utils.get_file("eval.json", 
# "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
# "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")
# with open(train_path) as f: raw_train_data = json.load(f)
# with open(eval_path) as f: raw_eval_data = json.load(f)
