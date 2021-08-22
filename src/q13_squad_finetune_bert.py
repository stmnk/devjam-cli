import json
from q13_Q_A_fine_tuning import (
    create_squad_two_examples,
    create_inputs_targets,
    model,
    ValidationCallback
)


with open('squad_2/train-v2.0.json') as f: 
    raw_train_data = json.load(f)
train_squad_examples = create_squad_two_examples(raw_train_data)

# print(vars(train_squad_examples[0]))

tuple_of_input_targets_list = create_inputs_targets(train_squad_examples)
x_train, y_train = tuple_of_input_targets_list

# print('x_train')
# print(x_train)

# print('y_train')
# print(y_train)

print(f"{len(train_squad_examples)} training points created.")


with open('squad_2/dev-v2.0.json') as f: 
    raw_eval_data = json.load(f)
eval_squad_examples = create_squad_two_examples(raw_eval_data)

x_eval, y_eval = create_inputs_targets(eval_squad_examples)

print(f"{len(eval_squad_examples)} evaluation points created.")


model.fit(
    x_train, y_train, 
    epochs=3,
    batch_size=8, 
    callbacks=[
        ValidationCallback(
            x_eval, y_eval, 
            eval_squad_examples
        )
    ]
)

model.save_weights("./qa_weights.h5")

# model.save_model('./bert_squad2.h5')
