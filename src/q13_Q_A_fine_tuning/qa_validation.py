import re
import string
import numpy as np
from tensorflow import keras
from typing import List, Tuple, Any


class ValidationCallback(keras.callbacks.Callback):
    def __init__(
        self, 
        x_eval: List[List[List[int]]], 
        y_eval: List[List[int]],
        eval_squad_examples: List[Any]
    ):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.eval_squad_examples = eval_squad_examples

    def normalize_text(self, text: string):
        text = text.lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def on_epoch_end(self, epoch: int, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        eval_examples_no_skip = [_ for _ in self.eval_squad_examples if _.skip == False]
        
        predictions: List[Tuple[int, int]] = zip(pred_start, pred_end)
        
        count = 0
        for index, (start, end) in enumerate(predictions):
            squad_eg = eval_examples_no_skip[index]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            
            if start >= len(offsets):
                continue
            
            pred_char_start = offsets[start][0]
            
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]
            normalized_pred_ans = self.normalize_text(pred_ans)
            normalized_true_ans = [self.normalize_text(_) for _ in squad_eg.all_answers]
            
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        
        acc = count / len(self.y_eval[0])
        
        print(f"\nepoch={epoch + 1}, exact match score={acc:.2f}")

