from typing import List

import BERTSimilarity.BERTSimilarity as bertsimilarity


# Needed to silence BERT warning messages, see: https://stackoverflow.com/questions/67546911/python-bert-error-some-weights-of-the-model-checkpoint-at-were-not-used-when # noqa
from transformers import logging
logging.set_verbosity_error()


def force_choice_with_bert(text: str, choices: List[str]):
    bertsim = bertsimilarity.BERTSimilarity()

    top_score = -float('inf')
    top_choice = None
    for choice in choices:
        score = bertsim.calculate_distance(text, choice)

        if score > top_score:
            top_score = score
            top_choice = choice

    return top_choice
