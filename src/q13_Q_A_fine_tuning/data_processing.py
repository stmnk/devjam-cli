import tokenizers
import numpy as np
import tensorflow_hub as hub
from typing import Optional, List, Tuple, Dict
from tokenizers import BertWordPieceTokenizer


class QADataSample:
    def __init__(
        self, 
        question: str, 
        context: str, 
        start_char_idx: Optional[int]=None, 
        answer_text: Optional[str]=None, 
        all_answers: Optional[List[str]]=None, 
        max_seq_length: int=384,
        tokenizer: BertWordPieceTokenizer=BertWordPieceTokenizer(
            vocab=hub.KerasLayer(
                "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", 
                trainable=True
            ).resolved_object.vocab_file.asset_path.numpy().decode("utf-8"), 
            lowercase=True
        ),
    ):
        self.question = question
        self.context = context
        self.start_character_index = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip: bool = False
        self.start_token_idx: int = -1
        self.end_token_idx: int = -1
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def preprocess(self):
        context: str = " ".join(str(self.context).split())
        tokenized_context: tokenizers.Encoding = self.tokenizer.encode(context)

        question: str = " ".join(str(self.question).split())
        tokenized_question: tokenizers.Encoding = self.tokenizer.encode(question)


        if self.answer_text is not None:
            answer: str = " ".join(str(self.answer_text).split())
            end_character_index: int = self.start_character_index + len(answer)
            
            if end_character_index >= len(context):
                self.skip = True
                return
            
            list_of_chars_in_answer: List[int] = [0] * len(context)
            for index in range(self.start_character_index, end_character_index):
                list_of_chars_in_answer[index] = 1

            list_of_answer_token_indices: List[int] = []
            context_offsets_list: List[Tuple[int, int]] = tokenized_context.offsets
            for index, (start, end) in enumerate(context_offsets_list):
                if sum(list_of_chars_in_answer[start:end]) > 0:
                    list_of_answer_token_indices.append(index)
            
            if len(list_of_answer_token_indices) == 0:
                self.skip = True
                return
            
            self.start_token_idx = list_of_answer_token_indices[0]
            self.end_token_idx = list_of_answer_token_indices[-1]

        input_ids: tokenizers.Encoding = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids: List[int] = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask: List[int] = [1] * len(input_ids)
        padding_length: int = self.max_seq_length - len(input_ids)
        
        if padding_length > 0:
            input_ids: tokenizers.Encoding = input_ids + ([0] * padding_length)
            attention_mask: List[int] = attention_mask + ([0] * padding_length)
            token_type_ids: List[int] = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return

        self.input_word_ids: tokenizers.Encoding = input_ids
        self.input_type_ids: List[int] = token_type_ids
        self.input_mask: List[int] = attention_mask
        self.context_token_to_char: List[Tuple[int, int]] = tokenized_context.offsets

def create_squad_two_examples(raw_squad_two_data) -> List[QADataSample]:
    squad_examples: List = []
    for qa_item in raw_squad_two_data["data"]:
        paragraphs = qa_item["paragraphs"]
        for paragraph in paragraphs:
            context: str = paragraph["context"]
            question_answers = paragraph["qas"]
            for q_answer in question_answers:
                question: str = q_answer["question"]
                if "answers" in q_answer: 
                    answers_list: List[str] = q_answer["answers"]
                    if answers_list:
                        first_answer = answers_list[0]
                        answer_text: str = first_answer["text"]
                        all_answers = [_["text"] for _ in answers_list]
                        start_character_index = first_answer["answer_start"]
                        squad_two_example: QADataSample = QADataSample(
                            question, context, start_character_index, answer_text, all_answers
                        )
                else:
                    squad_two_example = QADataSample(question, context)
                squad_two_example.preprocess()
                squad_examples.append(squad_two_example)
    return squad_examples

def create_inputs_targets(squad_examples: List[QADataSample]) -> Tuple[List[List[List[int]]], List[List[int]]]:
    dataset_dictionary: Dict[str, List] = {
        "input_word_ids":  [],
        "input_type_ids":  [],
        "input_mask":      [],
        "start_token_idx": [],
        "end_token_idx":   [],
    }

    for squad_item in squad_examples:
        if squad_item.skip == False:
            for key in dataset_dictionary:
                dataset_dictionary[key].append(getattr(squad_item, key))

    for key in dataset_dictionary:
        dataset_dictionary[key] = np.array(dataset_dictionary[key])
    
    x = [
            dataset_dictionary["input_word_ids"],
            dataset_dictionary["input_mask"],
            dataset_dictionary["input_type_ids"]
        ]
    y = [
            dataset_dictionary["start_token_idx"], 
            dataset_dictionary["end_token_idx"]
        ]
    return x, y


""" 
A concrete illustration of SQAD QADataSample example instance: 

print(vars(train_squad_examples[0]))

first_squad_two_example = {
    'question': 'When did Beyonce start becoming popular?', 
    'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".', 
    'start_character_index': 269, 
    'answer_text': 'in the late 1990s', 
    'all_answers': ['in the late 1990s'], 
    'skip': False, 
    'start_token_idx': 67, 
    'end_token_idx': 70, 
    'max_seq_length': 384, 
    'tokenizer': Tokenizer(
        vocabulary_size=30522, 
        model=BertWordPiece, 
        unk_token=[UNK], 
        sep_token=[SEP], 
        cls_token=[CLS], 
        pad_token=[PAD], 
        mask_token=[MASK], 
        clean_text=True, 
        handle_chinese_chars=True, 
        strip_accents=None, 
        lowercase=True, 
        wordpieces_prefix=##
    ), 
    'input_word_ids': [101, 20773, 21025, 19358, 22815, 1011, 5708, 1006, 1013, 12170, 23432, 29715, 3501, 29678, 12325, 29685, 1013, 10506, 1011, 10930, 2078, 1011, 2360, 1007, 1006, 2141, 2244, 1018, 1010, 3261, 1007, 2003, 2019, 2137, 3220, 1010, 6009, 1010, 2501, 3135, 1998, 3883, 1012, 2141, 1998, 2992, 1999, 5395, 1010, 3146, 1010, 2016, 2864, 1999, 2536, 4823, 1998, 5613, 6479, 2004, 1037, 2775, 1010, 1998, 3123, 2000, 4476, 1999, 1996, 2397, 4134, 2004, 2599, 3220, 1997, 1054, 1004, 1038, 2611, 1011, 2177, 10461, 1005, 1055, 2775, 1012, 3266, 2011, 2014, 2269, 1010, 25436, 22815, 1010, 1996, 2177, 2150, 2028, 1997, 1996, 2088, 1005, 1055, 2190, 1011, 4855, 2611, 2967, 1997, 2035, 2051, 1012, 2037, 14221, 2387, 1996, 2713, 1997, 20773, 1005, 1055, 2834, 2201, 1010, 20754, 1999, 2293, 1006, 2494, 1007, 1010, 2029, 2511, 2014, 2004, 1037, 3948, 3063, 4969, 1010, 3687, 2274, 8922, 2982, 1998, 2956, 1996, 4908, 2980, 2531, 2193, 1011, 2028, 3895, 1000, 4689, 1999, 2293, 1000, 1998, 1000, 3336, 2879, 1000, 1012, 102, 2043, 2106, 20773, 2707, 3352, 2759, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'input_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'input_mask':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'context_token_to_char': [(0, 0), (0, 7), (8, 10), (10, 15), (16, 23), (23, 24), (24, 30), (31, 32), (32, 33), (33, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 42), (42, 43), (43, 44), (45, 48), (48, 49), (49, 51), (51, 52), (52, 53), (53, 56), (56, 57), (58, 59), (59, 63), (64, 73), (74, 75), (75, 76), (77, 81), (81, 82), (83, 85), (86, 88), (89, 97), (98, 104), (104, 105), (106, 116), (116, 117), (118, 124), (125, 133), (134, 137), (138, 145), (145, 146), (147, 151), (152, 155), (156, 162), (163, 165), (166, 173), (173, 174), (175, 180), (180, 181), (182, 185), (186, 195), (196, 198), (199, 206), (207, 214), (215, 218), (219, 226), (227, 239), (240, 242), (243, 244), (245, 250), (250, 251), (252, 255), (256, 260), (261, 263), (264, 268), (269, 271), (272, 275), (276, 280), (281, 286), (287, 289), (290, 294), (295, 301), (302, 304), (305, 306), (306, 307), (307, 308), (309, 313), (313, 314), (314, 319), (320, 327), (327, 328), (328, 329), (330, 335), (335, 336), (337, 344), (345, 347), (348, 351), (352, 358), (358, 359), (360, 366), (367, 374), (374, 375), (376, 379), (380, 385), (386, 392), (393, 396), (397, 399), (400, 403), (404, 409), (409, 410), (410, 411), (412, 416), (416, 417), (417, 424), (425, 429), (430, 436), (437, 439), (440, 443), (444, 448), (448, 449), (450, 455), (456, 462), (463, 466), (467, 470), (471, 478), (479, 481), (482, 489), (489, 490), (490, 491), (492, 497), (498, 503), (503, 504), (505, 516), (517, 519), (520, 524), (525, 526), (526, 530), (530, 531), (531, 532), (533, 538), (539, 550), (551, 554), (555, 557), (558, 559), (560, 564), (565, 571), (572, 581), (581, 582), (583, 589), (590, 594), (595, 601), (602, 608), (609, 612), (613, 621), (622, 625), (626, 635), (636, 639), (640, 643), (644, 650), (650, 651), (651, 654), (655, 662), (663, 664), (664, 669), (670, 672), (673, 677), (677, 678), (679, 682), (683, 684), (684, 688), (689, 692), (692, 693), (693, 694), (0, 0)]
}

"""

"""
A concrete illustration of `create_inputs_targets` output for all SQAD QADataSample examples instances:
print('x_train')
print(x_train)

print('y_train')
print(y_train)


x_train
[
    array([   
        [  101, 20773, 21025, ...,     0,     0,     0],
        [  101, 20773, 21025, ...,     0,     0,     0],
        [  101, 20773, 21025, ...,     0,     0,     0],
        ...,
        [  101, 28045,  4956, ...,     0,     0,     0],
        [  101, 28045,  4956, ...,     0,     0,     0],
        [  101, 28045,  4956, ...,     0,     0,     0]
    ]),  # dataset_dictionary["input_word_ids"]
    array([
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        ...,
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]
    ]),  # dataset_dictionary["input_mask"]
    array([
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]
    ]),  # dataset_dictionary["input_type_ids"]
]

y_train
[
    array([ 67,  55, 128, ...,   1,   1,   1]),  # dataset_dictionary["start_token_idx"]
    array([ 70,  57, 128, ...,   3,   3,   3]),  # dataset_dictionary["end_token_idx"]
]
"""