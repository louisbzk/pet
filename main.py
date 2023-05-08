from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AlbertForSequenceClassification,
    AlbertForMaskedLM,
    AlbertTokenizer,
    AlbertConfig,
)
from transformers.data.metrics import simple_accuracy

from pet.wrapper import (
    WrapperConfig,
    TransformerModelWrapper,
    CONFIG_NAME,
    SEQUENCE_CLASSIFIER_WRAPPER,
    MLM_WRAPPER,
    PLM_WRAPPER,
    WRAPPER_TYPES,
    PREPROCESSORS,
    MODEL_CLASSES,
    EVALUATION_STEP_FUNCTIONS,
    TRAIN_STEP_FUNCTIONS,
)
from pet.preprocessor import Preprocessor
from pet.tasks import TASK_HELPERS
from pet.pvp import PVP
from pet.utils import (
    InputExample,
    InputFeatures,
    DictDataset,
    distillation_loss,
    exact_match,
)
from pet import EvalConfig

import log
import json
import jsonpickle
import os
from typing import List, Dict, Optional, Iterable

import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, Dataset
from tqdm import trange, tqdm


LOGGER = log.get_logger("root")


def concat_collate_fn(batch):
    return list(batch)


class InputExamplesDataset(Dataset):
    def __init__(self, input_examples):
        self.examples = input_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CustomTransformerWrapper(TransformerModelWrapper):
    def __init__(self, config: WrapperConfig):
        super().__init__(config)

    def _convert_example_to_features(self,
                                     example: InputExample,
                                     labelled: bool,
                                     priming: bool) -> InputFeatures:
        """
        Convert a single example to features.

        This is needed to avoid building multiple datasets at once, one for each PVP.
        Instead, the input examples are built individually.
        """
        input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming)
        if self.task_helper:
            self.task_helper.add_special_input_features(example, input_features)
        return input_features

    def generate_input_from_example(self,
                                    example: InputExample,
                                    labelled: bool = True,
                                    priming: bool = False) -> Dict[str, torch.Tensor]:
        features = self._convert_example_to_features(example, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor(features.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(features.attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(features.token_type_ids, dtype=torch.long),
            'labels': torch.tensor(features.label, dtype=torch.long),
            'mlm_labels': torch.tensor(features.mlm_labels, dtype=torch.long),
            'logits': torch.tensor(features.logits, dtype=torch.float),
            'idx': torch.tensor(features.idx, dtype=torch.long)
        }

        if self.task_helper:
            self.task_helper.add_features_to_single_feature_dict(features, feature_dict)

        return feature_dict


class MultiPVPPet:
    def __init__(self,
                 base_config: WrapperConfig,
                 pvp_pattern_ids: Iterable[int],
                 ):
        self.model_wrappers = []
        self.meta = []
        for i, pattern_id in enumerate(pvp_pattern_ids):
            config = base_config
            config.pattern_id = pattern_id
            self.model_wrappers.append(CustomTransformerWrapper(config))
            self.meta.append({
                "pattern_id": pattern_id,
            })
        self.n_models = len(self.model_wrappers)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.pvp_weights: nn.Parameter = torch.nn.Parameter(
            torch.zeros(self.n_models, dtype=torch.float32)
        ).to(self.device)

    @staticmethod
    def _init_train_dataloader(
            task_train_data,
            unlabeled_data,
            train_batch_size,
            unlabeled_batch_size,
            lm_training,
            use_logits,
            max_steps,
            num_train_epochs
    ) -> tuple[DataLoader, DataLoader, int]:
        train_dataset = InputExamplesDataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=train_batch_size,
                                      collate_fn=concat_collate_fn)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_dataset = InputExamplesDataset(unlabeled_data)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                              sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size,
                                              collate_fn=concat_collate_fn)

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if max_steps > 0:
            total_training_steps = max_steps
            num_train_epochs = max_steps // max(1, len(train_dataloader)) + 1
        else:
            total_training_steps = len(train_dataloader) * num_train_epochs

        return train_dataloader, unlabeled_dataloader, total_training_steps

    def _init_optimizers_schedulers(
            self,
            weight_decay,
            models_optimizer_lr,
            models_optimizer_eps,
            models_scheduler_warmup_steps,
            total_training_steps,
            pvp_weights_optimizer_lr,
            pvp_weights_optimizer_eps,
            pvp_weights_scheduler_warmup_steps
    ) -> tuple[
        torch.optim.AdamW,
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.AdamW,
        torch.optim.lr_scheduler.LambdaLR,
    ]:
        models_optimizer_params = []
        no_decay = ['bias', 'LayerNorm.weight']
        for wrapper in self.model_wrappers:
            models_optimizer_params += [
                {'params': [p for n, p in wrapper.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in wrapper.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        models_optimizer = torch.optim.AdamW(params=models_optimizer_params,
                                             lr=models_optimizer_lr,
                                             eps=models_optimizer_eps)
        models_scheduler = get_linear_schedule_with_warmup(models_optimizer,
                                                           num_warmup_steps=models_scheduler_warmup_steps,
                                                           num_training_steps=total_training_steps)

        pvp_weights_optimizer = torch.optim.AdamW(params=self.pvp_weights,
                                                  lr=pvp_weights_optimizer_lr,
                                                  eps=pvp_weights_optimizer_eps)
        pvp_weights_scheduler = get_linear_schedule_with_warmup(pvp_weights_optimizer,
                                                                num_warmup_steps=pvp_weights_scheduler_warmup_steps,
                                                                num_training_steps=total_training_steps)

        return models_optimizer, models_scheduler, pvp_weights_optimizer, pvp_weights_scheduler

    def _generate_batches_from_raw_batch(
            self,
            raw_batch: List[InputExample],
            labelled,
            priming,
    ) -> List[Dict[str, torch.Tensor]]:
        processed_batches = []
        for wrapper_idx, wrapper in enumerate(self.model_wrappers):
            features = wrapper._convert_examples_to_features(raw_batch, labelled=labelled, priming=priming)
            feature_dict = {
                'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
                'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                'labels': torch.tensor([f.label for f in features], dtype=torch.long),
                'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
                'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
                'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
            }
            if wrapper.config.wrapper_type == PLM_WRAPPER:
                feature_dict['perm_mask'] = torch.tensor([f.perm_mask for f in features], dtype=torch.float)
                feature_dict['target_mapping'] = torch.tensor([f.target_mapping for f in features], dtype=torch.float)

            if wrapper.task_helper:
                wrapper.task_helper.add_features_to_dict(features, feature_dict)

            processed_batches.append(feature_dict)

        return processed_batches

    def _prepare_unlabeled_batches(
            self,
            unlabeled_batches: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        for i, batch_dict in enumerate(unlabeled_batches):
            wrapper = self.model_wrappers[i]
            lm_input_ids = batch_dict['input_ids']
            batch_dict['input_ids'], batch_dict['mlm_labels'] = wrapper._mask_tokens(lm_input_ids)
            unlabeled_batches[i] = {k: t.to(self.device) for k, t in batch_dict.items()}

        return unlabeled_batches

    def _train_step(
            self,
            raw_batch: List[InputExample],
            unlabeled_dataloader,
            unlabeled_iter,
            lm_training,
            use_logits,
            alpha,
            temperature,
    ):
        for i in range(self.n_models):
            self.model_wrappers[i].model.train()

        # get batches for each model
        unlabeled_raw_batch = None
        batches = self._generate_batches_from_raw_batch(raw_batch=raw_batch, labelled=True, priming=False)
        for batch_dict in batches:
            for key, tensor in batch_dict.items():
                batch_dict[key] = tensor.to(self.device)

        if lm_training:
            while unlabeled_raw_batch is None:
                try:
                    unlabeled_raw_batch = unlabeled_iter.__next__()
                except StopIteration:
                    LOGGER.info("Resetting unlabeled dataset")
                    unlabeled_iter = unlabeled_dataloader.__iter__()
            unlabeled_batches = self._generate_batches_from_raw_batch(
                raw_batch=unlabeled_raw_batch, labelled=False, priming=False
            )
            unlabeled_batches = self._prepare_unlabeled_batches(unlabeled_batches)

            for batch_dict in unlabeled_batches:
                for key, tensor in batch_dict.items():
                    batch_dict[key] = tensor.to(self.device)

        losses = []
        for i, wrapper in enumerate(self.model_wrappers):
            batch = batches[i]
            unlabeled_batch = unlabeled_batches[i] if unlabeled_batches else None
            train_step_inputs = {
                'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                'use_logits': use_logits, 'temperature': temperature
            }
            loss = wrapper.task_helper.train_step(batch, **train_step_inputs) if wrapper.task_helper else None

            if loss is None:
                loss = TRAIN_STEP_FUNCTIONS[wrapper.config.wrapper_type](self)(batch, **train_step_inputs)

            losses.append(loss)

        return torch.cat(losses)

    def train(self,
              task_train_data: List[InputExample],
              train_batch_size: int = 8,
              num_train_epochs: int = 3,
              weight_decay: float = 0.0,
              models_optimizer_lr: float = 5e-5,
              models_optimizer_eps: float = 1e-8,
              models_scheduler_warmup_steps: int = 0,
              pvp_weights_optimizer_lr: float = 1e-4,
              pvp_weights_optimizer_eps: float = 1e-8,
              pvp_weights_scheduler_warmup_steps: int = 0,
              max_grad_norm: float = 1,
              logging_steps: int = 50,
              unlabeled_batch_size: int = 8,
              unlabeled_data: List[InputExample] = None,
              lm_training: bool = False,
              use_logits: bool = False,
              alpha: float = 0.8,
              temperature: float = 1,
              max_steps=-1,
              ):
        # TODO: create a base dataset (ok) and create features at each train step (todo).
        # NB: the functions to make 1 InputFeatures at a time might not be useful, it should create a batch
        train_dataloader, unlabeled_dataloader, total_training_steps = self._init_train_dataloader(
            task_train_data,
            unlabeled_data,
            train_batch_size,
            unlabeled_batch_size,
            lm_training,
            use_logits,
            max_steps,
            num_train_epochs
        )
        unlabeled_iter = unlabeled_dataloader.__iter__()
        # Initialize the models' optimizer: group the weights by those who should have decay or not
        (
            models_optimizer,
            models_scheduler,
            pvp_weights_optimizer,
            pvp_weights_scheduler,
        ) = self._init_optimizers_schedulers(
            weight_decay,
            models_optimizer_lr,
            models_optimizer_eps,
            models_scheduler_warmup_steps,
            total_training_steps,
            pvp_weights_optimizer_lr,
            pvp_weights_optimizer_eps,
            pvp_weights_scheduler_warmup_steps
        )

        current_step = 0
        tr_loss, logging_loss = 0., 0.
        for wrapper in self.model_wrappers:
            wrapper.model.zero_grad()
        epoch_itor = trange(num_train_epochs, desc="# Epoch")

        for _ in epoch_itor:
            batch_itor = tqdm(train_dataloader, desc="# Batch")
            for batch in batch_itor:
                losses = self._train_step(
                    raw_batch=batch,
                    unlabeled_dataloader=unlabeled_dataloader,
                    unlabeled_iter=unlabeled_iter,
                    lm_training=lm_training,
                    use_logits=use_logits,
                    alpha=alpha,
                    temperature=temperature,
                )

                models_optimizer.zero_grad()
                pvp_weights_optimizer.zero_grad()
                global_loss = torch.dot(torch.softmax(self.pvp_weights, dim=None), losses)
                global_loss.backward()
                for wrapper in self.model_wrappers:
                    torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), max_grad_norm)

                models_optimizer.step()
                models_scheduler.step()
                pvp_weights_optimizer.step()
                pvp_weights_scheduler.step()

                current_step += 1
                tr_loss += global_loss.item()
                if current_step % logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    learning_rate_scalar = models_scheduler.get_lr()
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{'step': current_step}}))

                if 0 < max_steps < current_step:
                    batch_itor.close()
                    break

            if 0 < max_steps < current_step:
                epoch_itor.close()
                break

        return current_step, (tr_loss / current_step if current_step > 0 else -1)

    def _eval_step(self,
                   raw_batch,
                   priming,
                   decoding_strategy):
        for wrapper in self.model_wrappers:
            wrapper.model.eval()

        batches = self._generate_batches_from_raw_batch(raw_batch=raw_batch, labelled=True, priming=priming)
        for batch_dict in batches:
            for key, tensor in batch_dict.items():
                batch_dict[key] = tensor.to(self.device)

        preds = []
        out_label_ids = []
        all_indices = []
        question_ids = []
        with torch.no_grad():
            for i, wrapper in enumerate(self.model_wrappers):
                batch = batches[i]
                labels = batch["labels"]
                indices = batch["idx"]

                if wrapper.task_helper:
                    logits = wrapper.task_helper.eval_step(batch, decoding_strategy=decoding_strategy)
                else:
                    logits = EVALUATION_STEP_FUNCTIONS[wrapper.config.wrapper_type](wrapper)(batch)

                preds.append(logits.detach().cpu().numpy())
                out_label_ids.append(labels.detach().cpu().numpy())
                all_indices.append(indices.detach().cpu().numpy())
                if "question_idx" in batch:
                    question_ids.append(batch["question_idx"].detach().cpu().numpy())

        return {
            "indices": np.array(all_indices),
            "logits": np.array(preds),
            "labels": np.array(out_label_ids),
            "question_ids": question_ids if question_ids else None
        }

    def eval(self,
             config: EvalConfig,
             eval_data: List[InputExample],
             priming_data: List[InputExample] | None,
             batch_size: int = 8,
             decoding_strategy: str = 'default'
             ) -> Dict:
        """
        Evaluate the underlying language model.

        :param config: the evaluation config
        :param eval_data: the evaluation examples to use
        :param priming_data: the priming data to use (if None, no priming)
        :param batch_size: the number of evaluation examples per batch and gpu
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')

        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """
        eval_dataset = InputExamplesDataset(eval_data)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

        priming = priming_data is not None
        if priming:
            for example in eval_data:
                example.meta["priming_data"] = priming_data

        # Do evaluation loop for each wrapper
        eval_result = {}
        for raw_batch in tqdm(eval_dataloader, desc="# Eval batch"):
            batch_result = self._eval_step(raw_batch, priming, decoding_strategy)
            if not eval_result:
                eval_result = batch_result
            else:
                for key, arr in eval_result.items():
                    for i in range(self.n_models):
                        eval_result[key][i] = np.concatenate([eval_result[key][i], batch_result[key][i]])

        # Process results for each wrapper
        metrics = config.metrics if config.metrics else ["acc"]
        final_results = {}
        for i, wrapper in enumerate(self.model_wrappers):
            predictions = np.argmax(eval_result["logits"][i], axis=1)
            labels = eval_result["labels"][i]
            question_ids = eval_result["question_ids"][i]
            scores = {}

            for metric in metrics:
                if metric == 'acc':
                    scores[metric] = simple_accuracy(predictions, labels)
                elif metric == 'f1':
                    scores[metric] = f1_score(labels, predictions)
                elif metric == 'f1-macro':
                    scores[metric] = f1_score(labels, predictions, average='macro')
                elif metric == 'em':
                    scores[metric] = exact_match(predictions, labels, question_ids)
                else:
                    raise ValueError(f"Metric '{metric}' not implemented")

            final_results[i] = {
                "scores": scores,
                "predictions": predictions
            }

        return final_results
