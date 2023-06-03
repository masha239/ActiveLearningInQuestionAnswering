import os
import pickle
import random

import pandas as pd
import torch
import torch.utils.data as data_utils
from transformers import EarlyStoppingCallback, GenerationConfig, DataCollatorWithPadding, Trainer, AutoTokenizer, \
    AutoModelForSequenceClassification, TrainingArguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.metrics import compute_metrics, compute_metrics_binary
from typing import NamedTuple
from datasets import Dataset
from src.extract_context import mean_pooling, cos
from tqdm import tqdm


def choose_best_pairs(probs, ids, part_ids):
    df = pd.DataFrame({'prob': probs, 'idx': ids, 'part_idx': part_ids})
    df = df.sort_values('prob', ascending=False).groupby('idx', as_index=False).first()
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


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, coef=0.5, binary=True):
        if dataset is not None:
            self.dataset = dataset
            self.binary = binary
            self.coef = coef
            self.doc_ids = list(set(dataset['document_id']))
            self.data = {doc_id: {'pos': [], 'neg': []} for doc_id in self.doc_ids}
            for idx, row in enumerate(dataset):
                if binary:
                    condition = row['labels'] == 1
                else:
                    condition = len(row['labels']) > 0

                if condition:
                    self.data[row['document_id']]['pos'].append(idx)
                else:
                    self.data[row['document_id']]['neg'].append(idx)
        else:
            self.dataset = None
            self.coef = coef
            self.ids = None
            self.data = None

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        if len(self.data[doc_id]['pos']) == 0:
            return self.dataset[random.choice(self.data[doc_id]['neg'])]
        if len(self.data[doc_id]['neg']) == 0:
            return self.dataset[random.choice(self.data[doc_id]['pos'])]
        if random.random() > self.coef:
            return self.dataset[random.choice(self.data[doc_id]['pos'])]
        else:
            return self.dataset[random.choice(self.data[doc_id]['neg'])]

    def filter_ids(self, doc_ids):
        new_dataset = TrainDataset()
        new_dataset.coef = self.coef
        new_dataset.dataset = self.dataset
        new_dataset.binary = self.binary
        doc_ids = list(set(doc_ids))
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in self.doc_ids]
        new_dataset.doc_ids = doc_ids
        new_dataset.data = {doc_id: self.data[doc_id] for doc_id in doc_ids}

        return new_dataset

    def filter_pairs(self, pairs):
        new_dataset = TrainDataset()
        new_dataset.coef = self.coef
        new_dataset.dataset = self.dataset
        new_dataset.binary = self.binary
        doc_ids = list(set([pair[0] for pair in pairs]))
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in self.doc_ids]
        new_dataset.doc_ids = doc_ids
        new_dataset.data = {doc_id: {'pos': [], 'neg': []} for doc_id in doc_ids}

        for doc_id in doc_ids:
            for idx in self.data[doc_id]['pos']:
                if (self.dataset[idx]['document_id'], self.dataset[idx]['part_id']) in pairs:
                    new_dataset.data[doc_id]['pos'].append(idx)
            for idx in self.data[doc_id]['neg']:
                if (self.dataset[idx]['document_id'], self.dataset[idx]['part_id']) in pairs:
                    new_dataset.data[doc_id]['neg'].append(idx)

        return new_dataset


class ActiveLearningData(NamedTuple):
    train_pool: Dataset
    train_dataset: TrainDataset
    train_bert: TrainDataset
    test_dataset: Dataset
    test_bert: Dataset
    val_pool: Dataset
    val_dataset: Dataset
    val_bert: Dataset
    val_answers: Dataset


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
            load_best_model_at_end=True,
            save_total_limit=4,
            eval_delay=self.config['eval_delay'],
            metric_for_best_model='EM',
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
            load_best_model_at_end=True,
            logging_dir='logs',
            report_to="none",
            push_to_hub=False,
            save_total_limit=4,
            eval_delay=self.config['eval_delay'],
            metric_for_best_model='roc_auc',
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
            max_length=self.config['max_length'],
        )

        if self.config['max_length'] == 32:
            patience_cnt = 5
        else:
            patience_cnt = 3

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience_cnt)],
        )

        self.model_binary = AutoModelForSequenceClassification.from_pretrained(
            self.config['checkpoint_binary'],
            num_labels=2
        ).to(self.config['device'])
        self.tokenizer_binary = AutoTokenizer.from_pretrained(self.config['checkpoint_binary'])
        self.data_collator_binary = DataCollatorWithPadding(tokenizer=self.tokenizer_binary, padding='max_length')

        self.trainer_binary = Trainer(
            model=self.model_binary,
            args=self.training_args_binary,
            tokenizer=self.tokenizer_binary,
            data_collator=self.data_collator_binary,
            compute_metrics=compute_metrics_binary,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience_cnt)],
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
        new_dataset = Dataset.from_dict(
            {
            'input_ids': dataset['input_ids'],
            'labels': predictions
            }
        )
        dataloader = self.trainer.get_eval_dataloader(new_dataset)
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch['input_ids'].to(self.config['device']), batch['labels'].to(self.config['device'])
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

    def _filter_pool(self, pool, bert):
        bert_probs = self._predict_probs_binary(bert)
        pairs = choose_best_pairs(bert_probs, bert['document_id'], bert['part_id'])
        pool = pool.filter(lambda sample: (sample['document_id'], sample['part_id']) in pairs)
        return pool

    def extract_embeddings(self, dataset):
        embeddings_list = []
        dataloader = self.trainer_binary.get_eval_dataloader(dataset)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs, masks = batch['input_ids'].to(self.config['device']), batch['attention_mask'].to(
                    self.config['device'])
                last_hidden_states = self.model_binary(inputs, masks, output_hidden_states=True).hidden_states[
                    -1].detach().cpu()
                embeddings_step = mean_pooling(last_hidden_states, batch['attention_mask'])
                embeddings_list.append(embeddings_step)

        embeddings = torch.concat(embeddings_list, axis=0)
        return embeddings

    def choose_best_idds(self, train_dataset, pool_dataset, best_ids_cnt, save_path=None, step=None):
        train_embeddings = self.extract_embeddings(train_dataset)
        pool_embeddings = self.extract_embeddings(pool_dataset)
        n = pool_embeddings.shape[0]
        m = train_embeddings.shape[0]
        coef = self.config['idds_coef']
        scores = []
        for i in tqdm(range(n)):
            pool_sum = sum([cos(pool_embeddings[i, :], pool_embeddings[j, :]).item() for j in range(n)])
            train_sum = sum([cos(pool_embeddings[i, :], train_embeddings[j, :]).item() for j in range(m)])
            score = coef * pool_sum / n - (1 - coef) * train_sum / m
            scores.append((pool_dataset['document_id'][i], score))

        scores.sort(key=lambda x: -x[1])
        df = pd.DataFrame({'doc_id': [x[0] for x in scores], 'score': [x[1] for x in scores]})
        if save_path is not None:
            df.to_csv(os.path.join(save_path, f'scores_{step}.csv'))
        return sorted([x[0] for x in scores[:best_ids_cnt]])

    def evaluate(self, val_pool, val_answers, val_bert, val):
        pool = self._filter_pool(val_pool, val_bert)
        predictions = self.trainer.predict(pool.remove_columns('labels')).predictions
        assert pool['document_id'] == val_answers['document_id']

        full = self.trainer.compute_metrics((predictions, val_answers['labels']), multilabel=True, calc_all=True)
        answer = self.trainer.evaluate(val)
        binary = self.trainer_binary.evaluate(val_bert)

        metrics = {'full_task': full, 'part_answer': answer, 'part_binary': binary}
        return metrics

    def predict(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.config['device'])
        with torch.no_grad():
            labels = self.model.generate(input_ids)
        return self.tokenizer.decode(labels, skip_special_tokens=True)[0]

    def _best_ids_from_probs(self, doc_ids, probs, best_ids_cnt):
        if self.config['unsertainty_strategy'] == 'min_normalized_prob':
            df = pd.DataFrame({'doc_id': doc_ids, 'prob': probs})
            df = df.sort_values('prob', ascending=False)
            df = df.groupby('doc_id', as_index=False).first().sort_values('prob', ascending=True)
            return df['doc_id'].values.tolist()[: best_ids_cnt]
        else:
            raise ValueError(f"Unsupported unsertainty strategy {self.config['unsertainty_strategy']}")

    def filter_bert_best(self, bert):
        bert_probs = self._predict_probs_binary(bert)
        pairs = choose_best_pairs(bert_probs, bert['document_id'], bert['part_id'])
        bert_filtered = bert.filter(lambda sample: (sample['document_id'], sample['part_id']) in pairs)
        return bert_filtered

    def _choose_ids(self, data, ids_in_train, strategy, save_path=None, step=None):
        document_ids = set(data.train_dataset.doc_ids)
        document_ids = list(document_ids - ids_in_train)

        random_ids_cnt = int(self.config['step_document_cnt'] * self.config['random_sample_fraction'])
        best_ids_cnt = self.config['step_document_cnt'] - random_ids_cnt

        if random_ids_cnt > 0:
            random_ids = set(random.sample(document_ids, min(len(document_ids), random_ids_cnt)))
        else:
            random_ids = set()

        if best_ids_cnt == 0:
            return random_ids

        document_ids = list(set(document_ids) - random_ids)
        pool_ids = set(random.sample(document_ids, min(len(document_ids), self.config['pool_document_cnt'])))

        if strategy == 'binary':
            bert_step = data.train_bert.dataset.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            bert_probs = self._predict_probs_binary(bert_step)
            best_ids = self._best_ids_from_probs(bert_step['document_id'], bert_probs, best_ids_cnt)

        elif strategy == 'answers':
            pool_step = data.train_pool.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            probs = self._predict_probs(pool_step)['prob']
            best_ids = self._best_ids_from_probs(pool_step['document_id'], probs, best_ids_cnt)

        elif strategy == 'binary+answers':
            bert_step = data.train_bert.dataset.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            pool_step = self._filter_pool(data.train_pool, bert_step)
            probs = self._predict_probs(pool_step)['prob']
            best_ids = self._best_ids_from_probs(pool_step['document_id'], probs, best_ids_cnt)
            if save_path is not None:
                with open(os.path.join(save_path, f'filtered_ids_{step}.pkl'), 'wb') as f:
                    pickle.dump({f'filtered_ids': pool_step['document_id']}, f)
                with open(os.path.join(save_path, f'probs_{step}.pkl'), 'wb') as f:
                    pickle.dump({f'probs': probs}, f)

        elif strategy == 'binary+idds':
            bert_step = data.train_bert.dataset.filter(lambda x: x['document_id'] in pool_ids).remove_columns('labels')
            bert_step_filtered = self.filter_bert_best(bert_step)
            bert_in_train = data.train_bert.dataset.filter(lambda x: x['document_id'] in ids_in_train).remove_columns('labels')
            bert_in_train_filtered = self.filter_bert_best(bert_in_train)
            best_ids = self.choose_best_idds(bert_in_train_filtered, bert_step_filtered, best_ids_cnt, save_path, step)

        else:
            raise ValueError(f'Unsupported strategy {strategy}')

        if save_path is not None:
            with open(os.path.join(save_path, f'best_ids_{step}.pkl'), 'wb') as f:
                pickle.dump({f'best_ids': best_ids}, f)
        return random_ids.union(set(best_ids))

    def _train_loop(self, data, ids_in_train, step, save_path=None, retrain=True):
        if retrain:
            self._reset_models()

        train_step = data.train_dataset.filter_ids(ids_in_train)
        train_binary_step = data.train_bert.filter_ids(ids_in_train)

        train_metrics = self.train(train_step, data.test_dataset)
        train_binary_metrics = self.train_binary(train_binary_step, data.test_bert)
        val_metrics = self.evaluate(data.val_pool, data.val_answers, data.val_bert, data.val_dataset)
        metrics = {'train': train_metrics, 'train_binary': train_binary_metrics, 'val': val_metrics}
        print(val_metrics)

        if save_path is not None:
            with open(os.path.join(save_path, f'metrics_{step}.pkl'), 'wb') as f:
                pickle.dump({f'step {step} metrics': metrics}, f)
            with open(os.path.join(save_path, f'step.pkl'), 'wb') as f:
                pickle.dump({f'step': step}, f)
            torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
            torch.save(self.model_binary.state_dict(), os.path.join(save_path, 'model_binary.pt'))

    def emulate_active_learning(self, data: ActiveLearningData, strategy, save_path=None, retrain=True):
        document_ids = list(set(data.train_dataset.doc_ids))

        if save_path is not None and 'step.pkl' in os.listdir(save_path):
            with open(os.path.join(save_path, f'step.pkl'), 'rb') as f:
                step = pickle.load(f)['step']
            with open(os.path.join(save_path, f'ids.pkl'), 'rb') as f:
                ids_in_train = pickle.load(f)['ids']
            self.model.load_state_dict(torch.load(os.path.join(save_path, 'model.pt')))
            self.model_binary.load_state_dict(torch.load(os.path.join(save_path, 'model_binary.pt')))
        else:
            step = 0
            ids_in_train = set(random.sample(document_ids, min(len(document_ids), self.config['start_document_cnt'])))
            print(f'Step {step}: {len(ids_in_train)} / {len(document_ids)} indexes are in train')
            self._train_loop(data, ids_in_train, step, save_path)
            if save_path is not None:
                with open(os.path.join(save_path, f'ids.pkl'), 'wb') as f:
                    pickle.dump({f'ids': ids_in_train}, f)

        while step < self.config['active_learning_steps_cnt']:
            step += 1
            print(f'Step {step}: choosing ids for train')
            ids_to_add = self._choose_ids(data, ids_in_train, strategy, save_path, step)
            ids_in_train = ids_in_train.union(ids_to_add)

            print(f'Step {step}: {len(ids_in_train)} / {len(document_ids)} indexes are in train')
            if retrain:
                self._train_loop(data, ids_in_train, step, save_path, retrain)
            else:
                self._train_loop(data, ids_to_add, step, save_path, retrain)
            if save_path is not None:
                with open(os.path.join(save_path, f'ids.pkl'), 'wb') as f:
                    pickle.dump({f'ids': ids_in_train}, f)
