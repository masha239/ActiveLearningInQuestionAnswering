import random

import evaluate
import numpy as np
import torch
from transformers import T5Tokenizer


def seed_everything(seed_val=0):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
exact_match = evaluate.load("exact_match")
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
roc_auc_score = evaluate.load("roc_auc")

tokenizer = T5Tokenizer.from_pretrained('t5-small', padding=True)


def compute_metrics_binary(eval_pred):
    predictions, labels = eval_pred
    result = dict()
    result['accuracy'] = accuracy.compute(predictions=np.argmax(predictions, axis=1), references=labels)['accuracy']
    result['roc_auc'] = roc_auc_score.compute(references=labels, prediction_scores=predictions[:, 1])['roc_auc']
    result['f1_micro'] = f1.compute(predictions=predictions[:, 1], references=labels, average="micro")['f1']
    result['f1_macro'] = f1.compute(predictions=predictions[:, 1], references=labels, average="macro")['f1']
    return {k: round(v, 4) for k, v in result.items()}


def compute_metrics(eval_pred, multilabel=False, calc_all=True):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    if not multilabel:
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    else:
        decoded_labels = [tokenizer.batch_decode(l, skip_special_tokens=True) for l in labels]

    result = dict()
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    result['rouge1'] = rouge_result['rouge1']
    result['rouge2'] = rouge_result['rouge2']

    if calc_all:
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result['Bleu'] = bleu_result['bleu']

        if not multilabel:
            em_result = exact_match.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                regexes_to_ignore=["the "],
                ignore_case=True,
                ignore_punctuation=True
            )
            result['EM'] = em_result['exact_match']
        else:
            em_results = []
            for pred, doc_labels in zip(decoded_preds, decoded_labels):
                max_em_result = 0
                for label in doc_labels:
                    em_result = exact_match.compute(
                        predictions=[pred],
                        references=[label],
                        ignore_case=True,
                        ignore_punctuation=True
                    )
                    max_em_result = max(max_em_result, em_result['exact_match'])
                em_results.append(max_em_result)
            result['EM'] = np.mean(em_results)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}