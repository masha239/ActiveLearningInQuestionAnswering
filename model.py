import pickle
import random

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
exact_match = evaluate.load("exact_match")
accuracy = evaluate.load("accuracy")


class ActiveQA:
    def __init__(self, config):
        self.config = config
        self._reset_models()
        self.model_is_trained = False

    def _reset_models(self):
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config['checkpoint_answer']
        ).to(self.config['device'])
        self.tokenizer = T5Tokenizer.from_pretrained(self.config['checkpoint_answer'], padding=True)
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.config['max_length']
        )

    def load_from_disk(self, path):
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save_to_disk(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def _get_compute_metrics(self):
        def compute_metrics(eval_pred, multilabel=False, calc_all=False):
            predictions, labels = eval_pred

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            if not multilabel:
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            else:
                decoded_labels = [self.tokenizer.batch_decode(l, skip_special_tokens=True) for l in labels]

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

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)

            return {k: round(v, 4) for k, v in result.items()}

        return compute_metrics

    def train(self, train_dataset, test_dataset):
        training_args = Seq2SeqTrainingArguments(
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
            metric_for_best_model='rouge2',
            load_best_model_at_end=True
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._get_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self.trainer.train()
        with open(self.config['log_path'], 'wb') as f:
            pickle.dump({'log': self.trainer.state.log_history}, f)
        self.models_is_trained = True

    def _get_probs_from_logits(self, logits, labels, normalized=True):
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

    def _predict_probs(self, dataset, normalized=True):
        predictions = self.trainer.predict(dataset).predictions
        dataloader = data_utils.DataLoader(predictions, batch_size=self.config['per_device_eval_batch_size'],
                                           shuffle=False)
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.config['device'])
                labels = self.model.generate(inputs)
                logits = self.model(input_ids=inputs, labels=labels).logits
                probs_list += self._get_probs_from_logits(logits, labels, normalized)
                labels_list += labels
        labels = [l.cpu().numpy() for l in labels_list]
        return {'prob': probs_list, 'labels': labels}

    def evaluate(self, test_dataset, test_text):
        res_dict = self._predict_probs(test_dataset)
        res_dict['labels'] = res_dict['labels']
        res_dict['document_id'] = test_dataset['document_id']
        df = pd.DataFrame(res_dict)
        df = df.sort_values('prob', ascending=False).groupby('document_id', as_index=False).first()
        df = df.sort_values('document_id')
        metrics = self.trainer.compute_metrics((df['labels'], test_text['answers']), multilabel=True, calc_all=True)
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

    def _choose_ids(self, dataset):
        document_ids = list(set(dataset['document_id']))

        random_ids_cnt = int(self.config['step_document_cnt'] * self.config['random_sample_fraction'])
        best_ids_cnt = self.config['step_document_cnt'] - random_ids_cnt

        random_ids = set(random.sample(document_ids, min(len(document_ids), random_ids_cnt)))
        document_ids = list(set(document_ids).difference(random_ids))

        pool_ids = set(random.sample(document_ids, min(len(document_ids), self.config['pool_document_cnt'])))
        filtered_dataset = dataset.filter(lambda x: x['document_id'] in pool_ids)
        filtered_dataset = filtered_dataset.map(self._predict_probs)
        return self._best_ids_from_probs(filtered_dataset['document_id'], filtered_dataset['prob'], best_ids_cnt)

    def emulate_active_learning(self, train_dataset, test_dataset, val_dataset, val_answers):
        document_ids = list(set(train_dataset['document_id']))
        ids_in_train = set(random.sample(document_ids, min(len(document_ids), self.config['start_document_cnt'])))

        train_filtered = train_dataset.filter(lambda x: x['document_id'] in ids_in_train and x['labels'] is not None)
        step_metrics = self.train(train_filtered, test_dataset)

        metrics = {'train': step_metrics}
        metrics['val'] = [self.evaluate(val_dataset)]

        for step in range(self.config['active_learning_steps_cnt']):
            print(f'step {step + 1}')
            ids_to_add = self._choose_ids(train_dataset.filter(lambda x: x['document_id'] not in ids_in_train))
            ids_in_train = ids_in_train.union(set(ids_to_add))

            train_filtered = train_dataset.filter(lambda x: x['document_id'] in ids_in_train and x['labels'] is not None)
            metrics['train'].append(self.train(train_filtered, test_dataset))
            metrics['val'].append(self.evaluate(val_dataset))

        return metrics