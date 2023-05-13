import json

from datasets import Dataset
from constants import PUNCTUATION_SET_TO_EXCLUDE


def format_answer(answer):
    alias = answer.replace('_', ' ').lower()
    alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
    return ' '.join(alias.split()).strip() + ' </s>'


def check_answer(part, answer, answer_type):
    if answer_type == 'short':
        return answer in part
    return answer[: int(0.75 * len(answer))] in part or answer[int(0.75 * len(answer)):] in part


def tokenize(text, tokenizer):
    return tokenizer(text, max_length=512, truncation=True)


def extract_tokenize(context, question, given_answers, answer_type, part_len, step, tokenizer):
    sentences = context.split('.')
    parts = ['.'.join(sentences[idx: idx + part_len]) for idx in range(0, len(sentences), step)]
    answers = []

    for part in parts:
        has_answer = 0
        new_part = f'question: {question}  context: {context} </s>'
        for answer in given_answers:
            if check_answer(part, answer, answer_type):
                has_answer = 1
                if answer_type == 'short':
                    answer = format_answer(answer)
                answers.append((tokenize(new_part, tokenizer), tokenize(answer, tokenizer).input_ids))
                break
        if has_answer == 0:
            answers.append((tokenize(new_part, tokenizer), None))

    assert len(answers) == len(parts)
    return answers


def prepare_dataset(dataset_path, tokenizer, part_len=15, step=10, max_context_len=50000, save_path=None,
                    answer_type='short_answers'):
    keys = ['input_ids', 'attention_mask', 'labels', 'document_id']
    dataset_dict = {key: [] for key in keys}

    with open(dataset_path, 'r') as json_file:
        doc_id = 0
        while True:
            doc_str = json_file.readline()
            if not doc_str:
                break

            doc = json.loads(doc_str)
            context = doc['context']
            question = doc['question']

            if len(context) > max_context_len:
                continue

            doc_id += 1

            if doc_id % 100 == 0:
                print(doc_id)

            if answer_type == 'short_answers' and len(doc['short_answers']) == 0:
                continue

            extracted_data = extract_tokenize(context, question, doc[answer_type], answer_type, part_len, step, tokenizer)
            for res in extracted_data:
                dataset_dict['document_id'].append(doc_id)
                dataset_dict['input_ids'].append(res[0].input_ids)
                dataset_dict['attention_mask'].append(res[0].attention_mask)
                dataset_dict['labels'].append(res[1])

    df = Dataset.from_dict(dataset_dict)
    if save_path is not None:
        df.save_to_disk(save_path)

    return df
