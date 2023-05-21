import gc
import pickle
import random

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from transformers import EarlyStoppingCallback, GenerationConfig, DataCollatorWithPadding, Trainer, AutoTokenizer, \
    AutoModelForSequenceClassification, TrainingArguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.metrics import compute_metrics, compute_metrics_binary
from typing import NamedTuple
from datasets import Dataset


def prepare_train_binary(dataset):
    k = np.mean(dataset['labels'])
    coef = k / (1 - k)
    return dataset.filter(lambda x: x['labels'] == 1 or random.random() < coef)


def choose_best_pairs(probs, ids, part_ids):
    df = pd.DataFrame({'prob': probs, 'idx': ids, 'part_idx': part_ids})
    df = df.sort_values('prob', ascending=False).groupby('idx', as_index=False).first()
    df = df.sort_values('idx')
    return [(document_id, part_idx) for (document_id, part_idx) in zip(df['idx'], df['part_idx'])]


def get_probs_from_logits(logits, labels, normalized=True):
    answer = []
    probs = torch.softmax(logits, -1)
    for sample_probs, sample_labels in zip(probs, labels):
        p = 1.0
        for idx, token in enumerate(sample_labels):
            p *= sample_probs[idx][token].item()
        if normalized:
            p = p ** (1 / len(sample_labels))
        answer.append(p)
    return answer


class ActiveLearningData(NamedTuple):
    train_pool: Dataset
    train_dataset: Dataset
    train_bert: Dataset
    test_dataset: Dataset
    test_bert: Dataset
    val_pool: Dataset
    val_dataset: Dataset
    val_bert: Dataset
    val_answers: Dataset


class TrainBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.coef = np.mean(dataset['labels'])
        self.ids = list(set(dataset['document_id']))
        self.data = {idx: {'pos': [], 'neg': []} for idx in self.ids}
        for row in dataset:
            if row['labels'] == 1:
                self.data[row['document_id']]['pos'].append(row)
            else:
                self.data[row['document_id']]['neg'].append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        if random.random() > self.coef and len(self.data[idx]['pos']) > 0:
            return random.choice(self.data[idx]['pos'])
        else:
            return random.choice(self.data[idx]['neg'])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


class ActiveQA:
    def __init__(self, config):
        self.config = config

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['model_output_dir'],
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['per_device_eval_batch_size'],
            weight_decay=self.config['weight_decay'],
            num_train_epochs=self.config['num_train_epochs'],
            predict_with_generate=True,
            generation_max_length=self.config['max_length_answer'],
            report_to="none",
            push_to_hub=False,
            logging_dir='logs',
            metric_for_best_model='rouge1',
            load_best_model_at_end=True,
            save_total_limit=3,
        )

        self.training_args_binary = TrainingArguments(
            output_dir="binary_model",
            learning_rate=self.config['learning_rate_binary'],
            per_device_train_batch_size=self.config['per_device_train_batch_size_binary'],
            per_device_eval_batch_size=self.config['per_device_eval_batch_size_binary'],
            num_train_epochs=self.config['num_train_epochs'],
            weight_decay=self.config['weight_decay_binary'],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            metric_for_best_model='roc_auc',
            load_best_model_at_end=True,
            logging_dir='logs',
            report_to="none",
            push_to_hub=False,
            save_total_limit=3,
        )

        self._reset_models()

        self.generation_config = GenerationConfig.from_pretrained(self.config['checkpoint_answer'])
        self.generation_config.max_length = self.config['max_length_answer']

    def _reset_models(self):
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config['checkpoint_answer']
        ).to(self.config['device'])
        self.tokenizer = T5Tokenizer.from_pretrained(self.config['checkpoint_answer'])
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.config['max_length']
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        self.model_binary = AutoModelForSequenceClassification.from_pretrained(
            self.config['checkpoint_binary'],
            num_labels=2
        ).to(self.config['device'])
        self.tokenizer_binary = AutoTokenizer.from_pretrained(self.config['checkpoint_binary'])
        self.data_collator_binary = DataCollatorWithPadding(tokenizer=self.tokenizer_binary)

        self.trainer_binary = Trainer(
            model=self.model_binary,
            args=self.training_args_binary,
            tokenizer=self.tokenizer_binary,
            data_collator=self.data_collator_binary,
            compute_metrics=compute_metrics_binary,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def load_from_disk(self, path):
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save_to_disk(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def train(self, train_dataset=None, test_dataset=None):
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = test_dataset
        self.trainer.train()

        return self.trainer.state.log_history

    def train_binary(self, train_dataset=None, test_dataset=None):
        self.trainer_binary.train_dataset = train_dataset
        self.trainer_binary.eval_dataset = test_dataset
        self.trainer_binary.train()

        return self.trainer_binary.state.log_history

    def _predict_probs(self, dataset, normalized=True):
        predictions = self.trainer.predict(dataset).predictions
        new_dataset = CustomDataset(
            torch.tensor([x + [0] * (self.config['max_length'] - len(x)) for x in dataset['input_ids']]),
            predictions
        )
        dataloader = data_utils.DataLoader(
            new_dataset,
            batch_size=self.config['per_device_eval_batch_size'],
            shuffle=False
        )
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.config['device']), labels.to(self.config['device'])
                logits = self.model(input_ids=inputs, labels=labels).logits
                probs = get_probs_from_logits(logits, labels, normalized)
                probs_list += probs
                labels_list += labels
        labels = [label.cpu().numpy() for label in labels_list]
        return {'prob': probs_list, 'labels': labels}

    def _predict_probs_binary(self, dataset):
        predictions = self.trainer_binary.predict(dataset).predictions
        predictions = torch.softmax(torch.tensor(predictions), -1).numpy()
        return predictions[:, 1]

    def evaluate(self, val_pool, val_answers, val_bert):
        bert_probs = self._predict_probs_binary(val_bert)
        pairs = choose_best_pairs(bert_probs, val_bert['document_id'], val_bert['part_id'])
        pool = val_pool.filter(lambda x: (x['document_id'], x['part_id']) in pairs)
        res_dict = self._predict_probs(pool.remove_columns('labels'))
        res_dict['document_id'] = pool['document_id']
        df = pd.DataFrame(res_dict)
        df = df.sort_values('document_id')
        metrics = self.trainer.compute_metrics((df['labels'], val_answers['labels']), multilabel=True, calc_all=True)
        return metrics

    def predict(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.config['device'])
        with torch.no_grad():
            labels = self.model.generate(input_ids)
        return self.tokenizer.decode(labels, skip_special_tokens=True)[0]

    def _best_ids_from_probs(self, doc_ids, probs, best_ids_cnt):
        if self.config['unsertainty_strategy'] == 'min_normalized_prob':
            df = pd.DataFrame({'doc_id': doc_ids, 'prob': probs})
            df = df.sort_values('prob', ascending=True)
            df = df.groupby('doc_id', as_index=False).first().sort_values('prob', ascending=True)
            return df['doc_id'].values.tolist()[: best_ids_cnt]
        else:
            raise ValueError(f"Unsupported unsertainty strategy {self.config['unsertainty_strategy']}")

    def _choose_ids(self, data, ids_in_train, strategy):
        document_ids = set(data.train_dataset['document_id'])
        document_ids = list(document_ids - ids_in_train)

        random_ids_cnt = int(self.config['step_document_cnt'] * self.config['random_sample_fraction'])
        best_ids_cnt = self.config['step_document_cnt'] - random_ids_cnt

        random_ids = set(random.sample(document_ids, min(len(document_ids), random_ids_cnt)))
        if best_ids_cnt == 0:
            return random_ids

        document_ids = list(set(document_ids) - random_ids)
        pool_ids = set(random.sample(document_ids, min(len(document_ids), self.config['pool_document_cnt'])))

        if strategy == 'binary':
            bert_step = data.train_bert.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            bert_probs = self._predict_probs_binary(bert_step)
            best_ids = self._best_ids_from_probs(bert_step['document_id'], bert_probs, best_ids_cnt)
        elif strategy == 'answers':
            pool_step = data.train_pool.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            probs = self._predict_probs(pool_step)['prob']
            best_ids = self._best_ids_from_probs(pool_step['document_id'], probs, best_ids_cnt)
        elif strategy == 'binary+answers':
            bert_step = data.train_bert.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            bert_probs = self._predict_probs_binary(bert_step)
            pairs = choose_best_pairs(bert_probs, bert_step['document_id'], bert_step['part_id'])
            pool_step = data.train_pool.filter(lambda x: (x['document_id'], x['part_id']) in pairs).remove_columns(
                'labels')
            probs = self._predict_probs(pool_step)['prob']
            best_ids = self._best_ids_from_probs(pool_step['document_id'], probs, best_ids_cnt)
        else:
            raise ValueError(f'Unsupported strategy {strategy}')

        return random_ids.union(set(best_ids))

    def emulate_active_learning(self, data: ActiveLearningData, strategy, save_path=None):

        document_ids = list(set(data.train_dataset['document_id']))
        ids_in_train = set(random.sample(document_ids, min(len(document_ids), self.config['start_document_cnt'])))

        print(f'Step 0: {len(ids_in_train)} / {len(document_ids)} indexes are in train')

        train_step = data.train_dataset.filter(lambda x: x['document_id'] in ids_in_train)
        train_binary_step = data.train_bert.filter(lambda x: x['document_id'] in ids_in_train)

        train_metrics = self.train(train_step, data.test_dataset)
        train_binary_metrics = self.train_binary(train_binary_step, data.test_bert)
        eval_metrics = self.evaluate(data.val_pool, data.val_answers, data.val_bert)
        print(eval_metrics)
        metrics = {'train': [train_metrics], 'train_binary': [train_binary_metrics], 'val': [eval_metrics]}

        del train_step
        del train_binary_step
        gc.collect()

        for step in range(self.config['active_learning_steps_cnt']):
            self._reset_models()

            print(f'Step {step + 1}: choosing ids for train')
            ids_to_add = self._choose_ids(data, ids_in_train, strategy)
            ids_in_train = ids_in_train.union(ids_to_add)

            print(f'Step {step + 1}: {len(ids_in_train)} / {len(document_ids)} indexes are in train')

            train_step = data.train_dataset.filter(lambda x: x['document_id'] in ids_in_train)
            train_binary_step = data.train_bert.filter(lambda x: x['document_id'] in ids_in_train)
            train_binary_step = prepare_train_binary(train_binary_step)

            train_metrics = self.train(train_step, data.test_dataset)
            train_binary_metrics = self.train_binary(train_binary_step, data.test_bert)

            eval_metrics = self.evaluate(data.val_pool, data.val_answers, data.val_bert)
            print(eval_metrics)
            metrics['train'].append(train_metrics)
            metrics['train_binary'].append(train_binary_metrics)
            metrics['val'].append(eval_metrics)

            del train_step
            del train_binary_step
            gc.collect()

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump({f'step {step + 1} metrics': metrics}, f)

        return metrics