from datasets import load_from_disk
from tokenize import extract_tokenize, tokenize, format_answer
from datasets import Dataset


def format_sample(sample):
    question = sample['question']['text']
    context = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    long_answers = sample['annotations']['long_answer']
    short_answers = sample['annotations']['short_answers']

    context_string = " ".join([context[i] for i in range(len(context)) if not is_html[i]])

    for answer in sample['annotations']['yes_no_answer']:
        if answer == 0 or answer == 1:
            return {"question": question, "context": context_string, "short_answers": [], "long_answers": [],
                    "category": "no" if answer == 0 else "yes"}

    short_targets = []
    for s in short_answers:
        short_targets.extend(s['text'])
        short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s['start_token'] == -1:
            continue
        answer = context[s['start_token']: s['end_token']]
        html = is_html[s['start_token']: s['end_token']]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "long_short" if len(short_targets + long_targets) > 0 else "null"
    return {"question": question, "context": context_string, "short_answers": short_targets,
            "long_answers": long_targets, "category": category}


def format_valudation_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_sample).remove_columns(["annotations", "document", "id"])
    dataset = dataset.filter(lambda x: (x['category'] == 'long_short')).remove_columns(["category"])
    dataset = dataset.filter(lambda x: len(x['context']) <= 50000)
    return dataset


def prepare_dataset(dataset, part_len=15, step=10, save_path=None, save_path_answers=None, answer_type='short_answers'):
    keys = ['input_ids', 'attention_mask', 'labels', 'document_id']
    dataset_dict = {key: [] for key in keys}
    answers_dict = {'document_id': [], 'answers': []}
    doc_id = 0

    for doc in dataset:
        context = doc['context']
        question = doc['question']

        doc_id += 1

        if doc_id % 10 == 0:
            print(doc_id)

        if len(doc[answer_type]) == 0:
            continue

        extracted_data = extract_tokenize(context, question, doc[answer_type], answer_type, part_len, step)
        for res in extracted_data:
            dataset_dict['document_id'].append(doc_id)
            dataset_dict['input_ids'].append(res[0].input_ids)
            dataset_dict['attention_mask'].append(res[0].attention_mask)
            dataset_dict['labels'].append(res[1])

        answers_dict['document_id'].append(doc_id)
        if answer_type == 'short':
            all_doc_answers = [tokenize(format_answer(answer)).input_ids for answer in doc[answer_type]]
        else:
            all_doc_answers = [tokenize(answer).input_ids for answer in doc[answer_type]]
        answers_dict['answers'].append(all_doc_answers)

    df = Dataset.from_dict(dataset_dict)
    if save_path is not None:
        df.save_to_disk(save_path)

    df_answers = Dataset.from_dict(answers_dict)
    if save_path_answers is not None:
        df_answers.save_to_disk(save_path_answers)

    return df, df_answers
