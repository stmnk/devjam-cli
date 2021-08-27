import numpy as np
from q13_Q_A_fine_tuning import (
    create_squad_two_examples,
    create_inputs_targets,
    model
)
from keras.models import load_model

what_the_faq = "https://share.streamlit.io/charlywargnier/what-the-faq/main/app.py"
url_for_wtfaq = "https://awoiaf.westeros.org/index.php/Varamyr"
paragraph_from_wtfaq = "Varamyr, better known as Varamyr Sixskins, is a member of the free folk, and a skinchanger who controls three wolves, a snow bear, a shadowcat, and an eagle.[3] He was known as Lump until he took the name Varamyr.[1] Varamyr is small, grey-faced, and bald, with round shoulders. He wears a shadowskin cloak.[4] His three wolves are named One Eye, Stalker and Sly.[1] The man who would eventually call himself Varamyr was born the second child of a wildling family beyond the Wall, and nicknamed 'Lump' by his older sister Meha per wildling tradition. As he was born a month premature, and often ill, his mother waited until he was four to name him properly, by which time it was too late and everyone in his village only called him Lump. When Lump was six, he was jealous of his two-year-old brother Bump, who was much more vigorous and healthy, and skinchanged into one of his family's dogs (Loptail, Sniff, or Growler) to kill him. When Lump's father came upon Bump's body, the dogs were sniffing around it. ",

data = {
    "data": [
        {
            "title": "ASOIAF Varamyr",
            "paragraphs": [
                {
                    "context": "Varamyr, better known as Varamyr Sixskins, is a member of the free folk, and a skinchanger who "
                               "controls three wolves, a snow bear, a shadowcat, and an eagle.[3] He was known as Lump until he "
                               "took the name Varamyr.[1] Varamyr is small, grey-faced, and bald, with round shoulders. He wears a "
                               "shadowskin cloak.[4] His three wolves are named One Eye, Stalker and Sly.[1] The man who would "
                               "eventually call himself Varamyr was born the second child of a wildling family beyond the Wall, "
                               "and nicknamed 'Lump' by his older sister Meha per wildling tradition. As he was born a month "
                               "premature, and often ill, his mother waited until he was four to name him properly, by which time "
                               "it was too late and everyone in his village only called him Lump. When Lump was six, "
                               "he was jealous of his two-year-old brother Bump, who was much more vigorous and healthy, "
                               "and skinchanged into one of his family's dogs (Loptail, Sniff, or Growler) to kill him. When "
                               "Lump's father came upon Bump's body, the dogs were sniffing around it. ",
                    "qas": [
                        {
                            "question": "What is Varamyr also known as?",  # Varamyr Sixskins
                            "id": "Q1"
                        },
                        {
                            "question": "What was Varamyr known as until he took the name Varamyr?",  # Lump
                            "id": "Q2"
                        },
                        {
                            "question": "What is Varamyr?",  # bald
                            "id": "Q3"
                        },
                        {
                            "question": "What type of cloak does Varamyr wear?",  # shadowskin cloak
                            "id": "Q4"
                        },
                        {
                            "question": "What are the three wolves in Varamyr's name?",  # One Eye, Stalker and Sly
                            "id": "Q5"
                        },
                        {
                            "question": "What was Varamyr's older sister's nickname?",  # Meha per wildling tradition
                            "id": "Q6"
                        },
                        {
                            "question": "Who was Lump jealous of when he was six?",  # Bump
                            "id": "Q7"
                        },
                        {
                            "question": "Where was Varamir born?",  # beyond the Wall
                            "id": "Q8"
                        }
                    ]
                }
            ]
        }
    ]
}

test_samples = create_squad_two_examples(data)
x_test, _ = create_inputs_targets(test_samples)

model.load_weights('./qa_weights.h5')
# model = load_model('./bert_squad2.h5')

pred_start, pred_end = model.predict(x_test)

for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    test_sample = test_samples[idx]
    offsets = test_sample.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    pred_ans = None

    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]

    if end < len(offsets):
        pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
    else:
        pred_ans = test_sample.context[pred_char_start:]

    print("Q: " + test_sample.question)
    print("A: " + pred_ans)

# aegon_paragraph_context = "The maesters of the Citadel who keep the histories of Westeros have used Aegon's Conquest as their touchstone for the past three hundred years. Births, deaths, battles, and other events are dated either AC (After the Conquest) or BC (Before the Conquest). True scholars know that such dating is far from precise. Aegon Targaryen's conquest of the Seven Kingdoms did not take place in a single day. More than two years passed between Aegon's landing and his Oldtown coronation…and even then the Conquest remained incomplete, since Dorne remained unsubdued. Sporadic attempts to bring the Dornishmen into the realm continued all through King Aegon's reign and well into the reigns of his sons, making it impossible to fix a precise end date for the Wars of Conquest. Even the start date is a matter of some misconception. Many assume, wrongly, that the reign of King Aegon I Targaryen began on the day he landed at the mouth of the Blackwater Rush, beneath the three hills where the city of King's Landing would eventually stand. Not so. The day of Aegon's Landing was celebrated by the king and his descendants, but the Conqueror actually dated the start of his reign from the day he was crowned and anointed in the Starry Sept of Oldtown by the High Septon of the Faith. This coronation took place two years after Aegon's Landing, well after all three of the major battles of the Wars of Conquest had been fought and won. Thus it can be seen that most of Aegon's actual conquering took place from 2–1 BC, Before the Conquest."
# aegon_paragraph_context_json = {
#     "paragraph_context": "The maesters of the Citadel who keep the histories of Westeros have used Aegon's "
#                          "Conquest as their touchstone for the past three hundred years. Births, deaths, "
#                          "battles, and other events are dated either AC (After the Conquest) or BC (Before the "
#                          "Conquest). True scholars know that such dating is far from precise. Aegon Targaryen's "
#                          "conquest of the Seven Kingdoms did not take place in a single day. More than two years "
#                          "passed between Aegon's landing and his Oldtown coronation…and even then the Conquest "
#                          "remained incomplete, since Dorne remained unsubdued. Sporadic attempts to bring the "
#                          "Dornishmen into the realm continued all through King Aegon's reign and well into the "
#                          "reigns of his sons, making it impossible to fix a precise end date for the Wars of "
#                          "Conquest. Even the start date is a matter of some misconception. Many assume, "
#                          "wrongly, that the reign of King Aegon I Targaryen began on the day he landed at the "
#                          "mouth of the Blackwater Rush, beneath the three hills where the city of King's "
#                          "Landing would eventually stand. Not so. The day of Aegon's Landing was celebrated "
#                          "by the king and his descendants, but the Conqueror actually dated the start of his "
#                          "reign from the day he was crowned and anointed in the Starry Sept of Oldtown by "
#                          "the High Septon of the Faith. This coronation took place two years after Aegon's "
#                          "Landing, well after all three of the major battles of the Wars of Conquest had been "
#                          "fought and won. Thus it can be seen that most of Aegon's actual conquering took "
#                          "place from 2–1 BC, Before the Conquest."}
