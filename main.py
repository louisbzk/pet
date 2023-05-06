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
from pet.pvp import PVP
from transformers import (
    InputExample,
    AdamW,
    get_linear_schedule_with_warmup,
    AlbertForSequenceClassification,
    AlbertForMaskedLM,
    AlbertTokenizer,
    AlbertConfig,
)

from pet.preprocessor import Preprocessor
from pet.tasks import TASK_HELPERS
from pet.utils import InputFeatures, DictDataset, distillation_loss
import log

import json
import jsonpickle
import os
from typing import List, Dict, Optional, Iterable
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, Dataset
from tqdm import trange, tqdm


LOGGER = log.get_logger("root")


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

    def train_step(
            self,
            batch: dict,
            unlabeled_iter,
            unlabeled_dataloader: DataLoader,
            device: torch.device,
            n_gpu: int = 1,
            gradient_accumulation_steps: int = 1,
            lm_training: bool = False,
            use_logits: bool = False,
            alpha: float = 0.8,
            temperature: float = 1,
            **_
    ) -> torch.Tensor:
        self.model.train()

        unlabeled_batch = None
        batch = {k: t.to(device) for k, t in batch.items()}

        if lm_training:
            while unlabeled_batch is None:
                try:
                    unlabeled_batch = unlabeled_iter.__next__()
                except StopIteration:
                    unlabeled_iter = unlabeled_dataloader.__iter__()

            lm_input_ids = unlabeled_batch['input_ids']
            unlabeled_batch['input_ids'], unlabeled_batch['mlm_labels'] = self._mask_tokens(lm_input_ids)
            unlabeled_batch = {k: t.to(device) for k, t in unlabeled_batch.items()}

        train_step_inputs = {
            'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
            'use_logits': use_logits, 'temperature': temperature
        }
        loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None

        if loss is None:
            loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        return loss

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
                                      batch_size=train_batch_size)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_dataset = InputExamplesDataset(unlabeled_data)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                              sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size)

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

    def _train_step(
            self,
            train_dataloader,
            unlabeled_dataloader,
            unlabeled_iter,
            lm_training,
            max_grad_norm,
            alpha,
            temperature,
            current_step,
            logging_steps,
    ):
        for i in range(self.n_models):
            self.model_wrappers[i].model.train()

        # get batches for each model
        # use _mask_tokens only ONCE (does not depend on pvp)
        unlabeled_batches = None
        batches = ...

        if lm_training:
            while unlabeled_batches is None:
                try:
                    unlabeled_batches = ...  # using unlabeled_iter
                except StopIteration:
                    LOGGER.info("Resetting unlabeled dataset")
                    unlabeled_iter = unlabeled_dataloader.__iter__()

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
        for i in range(self.n_models):
            self.model_wrappers[i].model.zero_grad()
        epoch_itor = trange(num_train_epochs, desc="# Epoch")

        for _ in epoch_itor:
            batch_itor = tqdm(train_dataloader, desc="# Batch")
            for batch in batch_itor:
                self._train_step(
                    train_dataloader,
                    unlabeled_dataloader,
                    unlabeled_iter,
                    lm_training,
                    max_grad_norm,
                    alpha,
                    temperature,
                    current_step,
                    logging_steps
                )

                current_step += 1
