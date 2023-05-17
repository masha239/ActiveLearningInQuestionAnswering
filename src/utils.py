import datetime
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split


def seed_everything(seed_val=0):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_data(df_path, answer_type, sample_cnt=None):
    df = pd.read_csv(df_path).drop('Unnamed: 0', axis=1)

    df = df[(df[answer_type].notna()) | (df['has_answer'] == False)]

    df['text_input'] = [f'question: {q}  context: {c} </s>' for q, c in zip(df['question'], df['context_prepared'])]
    df['label'] = [1 if x else 0 for x in df['has_answer']]

    df = df.drop(['question', 'context_prepared', 'has_answer'], axis=1)

    if sample_cnt is not None:
        doc_ids = random.sample(df['document_id'].unique().tolist(), sample_cnt)
    else:
        doc_ids = df['document_id'].unique().tolist()

    train_ids, test_ids = train_test_split(doc_ids, test_size=0.05)
    df_train = df[df.document_id.isin(train_ids)]
    df_test = df[df.document_id.isin(test_ids)]
    train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
    test_dataset = Dataset.from_pandas(df_test, preserve_index=False)
    return train_dataset, test_dataset


def create_config(device, model_dir=None):
    if model_dir is None:
        model_dir = f'qa_{int(datetime.datetime.now().timestamp())}'

    os.makedirs(model_dir)
    print(f'model_dir {model_dir} created')

    config = {
        'checkpoint_answer': 't5-small',
        'max_length': 512,
        'max_length_answer': 32,
        'learning_rate': 1e-5,
        'weight_decay': 1e-2,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 1,
        'device': device,
        'unsertainty_strategy': 'min_normalized_prob',
        'start_document_cnt': 1000,
        'active_learning_steps_cnt': 10,
        'pool_document_cnt': 2500,
        'step_document_cnt': 500,
        'random_sample_fraction': 1.0,
        'model_output_dir': os.path.join(model_dir, 'model'),
        'log_path': os.path.join(model_dir, 'logs.pkl')
    }

    with open(os.path.join(model_dir, 'config.yaml'), 'wb') as f:
        pickle.dump(config, f)

    return config


def get_train_test(dataset_path, test_size=0.05):
    dataset = load_from_disk('/kaggle/working/nq-short-20000-16-8-tokenized')
    dataset = dataset.filter(lambda x: x['labels'] is not None)
    train_ids, test_ids = train_test_split(list(set(dataset['document_id'])), test_size=test_size)
    train_dataset = dataset.filter(lambda x: x['document_id'] in train_ids)
    test_dataset = dataset.filter(lambda x: x['document_id'] in test_ids)
    return train_dataset, test_dataset


def get_val(val_path, answers_path, max_idx=None):
    dataset = load_from_disk(val_path)
    answers_dataset = load_from_disk(answers_path)
    if max_idx is not None:
        test_dataset = dataset.filter(lambda x: x['document_id'] < max_idx)
        test_answer_dataset = answers_dataset.filter(lambda x: x['document_id'] < max_idx)
    return test_dataset, test_answer_dataset
