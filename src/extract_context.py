import json
import pickle

import numpy as np
import torch
import torch.utils.data as data_utils
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from transformers import AutoTokenizer, AutoModel


def choose_best_context_bag_of_words(context, question, part_len=15, step=12, binary=True, ngram_range=(2, 2)):
    sentences = context.split('.')
    parts = ['.'.join(sentences[idx: idx + part_len]) for idx in range(0, len(sentences), step)]
    vectorizer = CountVectorizer(ngram_range=ngram_range, binary=binary).fit([question])
    parts_transformed = vectorizer.transform(parts).toarray()
    sums = np.sum(parts_transformed, axis=1)
    idx = np.argmax(sums)
    return parts[idx]


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


cos = nn.CosineSimilarity(dim=0)


def choose_best_context_embeddings(model, tokenizer, context, question, device, part_len=15, step=12):
    sentences = context.split('.')
    texts = ['.'.join(sentences[idx: idx + part_len]) for idx in range(0, len(sentences), step)] + [question]

    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    text_dataset = data_utils.TensorDataset(encoded_input['input_ids'], encoded_input['attention_mask'])
    text_dataloader = data_utils.DataLoader(text_dataset, batch_size=32, shuffle=False)

    token_embeddings_list = []
    with torch.no_grad():
        for inputs, masks in text_dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            model_output = model(inputs, masks)
            token_embeddings = model_output[0]
            token_embeddings_list.append(token_embeddings)

    token_embeddings = torch.concat(token_embeddings_list, axis=0)
    parts_embeddings = mean_pooling(token_embeddings, encoded_input['attention_mask'].to(device))
    scores = torch.tensor([cos(parts_embeddings[i, :], parts_embeddings[-1, :]) for i in range(len(texts) - 1)])
    idx = torch.argmax(scores)
    return texts[idx]


def extract_context(context, question, tokenizer=None, model=None, extractor_type='bag_of_words', binary=True,
                    ngram_range=(2, 2), part_len=15, step=12):
    if extractor_type == 'bag_of_words':
        context_extracted = choose_best_context_bag_of_words(
            context,
            question,
            part_len,
            step,
            binary,
            ngram_range
        )
    elif extractor_type == 'embeddings':
        context_extracted = choose_best_context_embeddings(
            model,
            tokenizer,
            context,
            question,
            part_len,
            step
        )

    elif extractor_type == 'full':
        context_extracted = context
    else:
        raise ValueError(f'Unsupported extractor type {extractor_type}')
    return context_extracted


def prepare_dataset(dataset_path, device, checkpoint=None, max_context_len=50000, extractor_type='bag_of_words', binary=True,
                    ngram_range=(2, 2), part_len=15, step=12, save_path=None, dataset_dict=None):
    keys = ['question', 'short_answers', 'long_answer']
    if dataset_dict is None:
        dataset_dict = {key: [] for key in keys}
        dataset_dict['context_prepared'] = []
        dataset_dict['lines_readed'] = 0
        start_line = 0
    else:
        start_line = dataset_dict['lines_readed']

    line_count = sum(1 for _ in open(dataset_path)) - start_line
    print(f'Total samples to prepare: {line_count}')

    tokenizer = None
    model = None
    if extractor_type == 'embeddings':
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True).to(device)

    with open(dataset_path, 'r') as json_file:
        n_str = 0
        while True:
            doc_str = json_file.readline()
            if not doc_str:
                break

            n_str += 1
            if n_str <= start_line:
                continue

            dataset_dict['lines_readed'] = n_str

            if n_str % 1000 == 0:
                print(f"{n_str - start_line} / {line_count} samples readed")
                with open('dictionary_dataset.pkl', 'wb') as f:
                    pickle.dump(dataset_dict, f)

            doc = json.loads(doc_str)
            context = doc['context']
            question = doc['question']

            if len(context) > max_context_len:
                continue

            for key in keys:
                dataset_dict[key].append(doc[key])

            context_extracted = extract_context(context, question, tokenizer, model, extractor_type, binary,
                                                ngram_range, part_len, step)
            dataset_dict['context_prepared'].append(context_extracted)

    del dataset_dict['lines_readed']
    dataset = Dataset.from_dict(dataset_dict)
    if save_path is not None:
        dataset.save_to_disk(save_path)

    return dataset
