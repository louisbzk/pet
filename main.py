import argparse
import log
import json
import jsonpickle
import os
from typing import List, Dict, Optional, Iterable
from pathlib import Path

import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, Dataset
from tqdm import trange, tqdm

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AlbertForSequenceClassification,
    AlbertForMaskedLM,
    AlbertTokenizer,
    AlbertConfig,
)
from transformers.data.metrics import simple_accuracy

from cli import load_pet_configs
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
from pet.tasks import TASK_HELPERS, PROCESSORS, TEST_SET, DEV_SET, load_examples, TRAIN_SET, UNLABELED_SET, METRICS, \
    DEFAULT_METRICS
from pet.pvp import CopaPVP
from pet.utils import (
    InputExample,
    InputFeatures,
    DictDataset,
    distillation_loss,
    exact_match, eq_div,
)
from pet import EvalConfig


LOGGER = log.get_logger("root")


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)


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


class PVPWeights(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(n, requires_grad=True, device=device))

    def forward(self, losses):
        return torch.dot(torch.softmax(self.weights, dim=0), losses)


class MultiPVPPet:
    """
    A wrapper around the TransformerModelWrapper models to use multiple PVPs at once.

    Warning: this trains multiple instance of Transformer models at once on GPU.
    """
    def __init__(self,
                 base_config: WrapperConfig,
                 pvp_pattern_ids: Iterable[int],
                 no_cuda: bool,
                 ):
        self.model_wrappers = []
        self.meta = []
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and not no_cuda else torch.device("cpu")
        for i, pattern_id in enumerate(pvp_pattern_ids):
            config = base_config
            config.pattern_id = pattern_id
            self.model_wrappers.append(CustomTransformerWrapper(config))
            self.model_wrappers[-1].model.to(self.device)
            self.meta.append({
                "pattern_id": pattern_id,
            })
        self.n_models = len(self.model_wrappers)
        self.pvp_weights = PVPWeights(self.n_models, self.device)

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

        pvp_weights_optimizer = torch.optim.AdamW(params=[self.pvp_weights.weights],
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
        for wrapper in self.model_wrappers:
            wrapper.model.zero_grad()

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
            wrapper.model.to(self.device)
            loss = wrapper.task_helper.train_step(batch, **train_step_inputs) if wrapper.task_helper else None

            if loss is None:
                loss = TRAIN_STEP_FUNCTIONS[wrapper.config.wrapper_type](wrapper)(batch, **train_step_inputs)

            losses.append(loss.reshape(1))

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
        epoch_itor = trange(num_train_epochs, desc="# Epoch")
        for wrapper in self.model_wrappers:
            wrapper.model.train()
        self.pvp_weights.train()

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
                global_loss = self.pvp_weights.forward(losses)
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

            epoch_itor.set_postfix({"loss": tr_loss / current_step})

            if 0 < max_steps < current_step:
                epoch_itor.close()
                break

        return current_step, (tr_loss / current_step if current_step > 0 else -1)

    def _eval_step(self,
                   raw_batch,
                   priming,
                   decoding_strategy):

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
            "indices": all_indices,
            "logits": preds,
            "labels": out_label_ids,
            "question_ids": question_ids if question_ids else None
        }

    def eval(self,
             metrics: List[str],
             eval_data: List[InputExample],
             priming_data: List[InputExample] | None,
             batch_size: int = 8,
             decoding_strategy: str = 'default'
             ) -> Dict:
        """
        Evaluate the underlying language model.

        :param metrics: the evaluation metrics
        :param eval_data: the evaluation examples to use
        :param priming_data: the priming data to use (if None, no priming)
        :param batch_size: the number of evaluation examples per batch and gpu
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')

        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """
        eval_dataset = InputExamplesDataset(eval_data)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=batch_size, collate_fn=concat_collate_fn)
        for wrapper in self.model_wrappers:
            wrapper.model.eval()
        self.pvp_weights.eval()

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
                    if arr is None:
                        continue
                    for i in range(self.n_models):
                        eval_result[key][i] = np.concatenate([arr[i], batch_result[key][i]])

        # Process results for each wrapper
        metrics = metrics if metrics else ["acc"]
        final_results = {}
        for i, wrapper in enumerate(self.model_wrappers):
            predictions = np.argmax(eval_result["logits"][i], axis=1)
            labels = eval_result["labels"][i]
            question_ids = eval_result["question_ids"][i] if eval_result["question_ids"] else None
            scores = {}

            for metric in metrics:
                if metric == 'acc':
                    scores[metric] = simple_accuracy(predictions, labels)
                elif metric == 'f1':
                    scores[metric] = f1_score(labels, predictions)
                elif metric == 'f1-macro':
                    scores[metric] = f1_score(labels, predictions, average='macro')
                elif metric == 'em' and question_ids is not None:
                    scores[metric] = exact_match(predictions, labels, question_ids)
                else:
                    raise ValueError(f"Metric '{metric}' not implemented")

            final_results[i] = {
                "pattern_id": self.meta[i]["pattern_id"],
                "scores": scores,
                "predictions": predictions,
            }

        final_results["pvp_weights"] = list(torch.softmax(self.pvp_weights.weights, dim=0).detach().cpu().numpy())

        return final_results


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--method", required=True, choices=['pet', 'ipet', 'sequence_classifier'],
                        help="The training method to use. Either regular sequence classification, PET or iPET.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
                             "for a permuted language model like XLNet (only for PET)")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--pet_per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--mpvp_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for the PVP weights")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--mpvp_adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer for the PVP weights")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--mpvp_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps for the PVP weights' scheduler")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")

    args = parser.parse_args()

    if args.task_name.lower() not in [
        "copa",
        "cb",
    ]:
        raise NotImplementedError(f"Task not supported by Multi-PVP PET: '{args.task_name.lower()}'")

    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    args.do_train = True
    args.do_eval = True

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    print(f"[INFO] Loading examples for task '{args.task_name}', \n"
          f"data_dir = '{args.data_dir}', \n"
          f"n_train_examples = {train_ex}, n_train_examples_per_label = {train_ex_per_label}, \n"
          f"n_test_examples = {test_ex}, n_test_examples_per_label = {test_ex_per_label}, \n"
          f"n_unlabeled_examples = {args.unlabeled_examples}")
    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    eval_data = load_examples(
        args.task_name, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
    unlabeled_data = load_examples(
        args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    # Fill unsupported args by MultiPVPPet, but are needed by the Config classes
    args.pet_gradient_accumulation_steps = 1
    args.verbalizer_file = None

    if len(args.pattern_ids) == 1:
        print(f"[WARNING] Only one pattern_id was provided: {args.pattern_ids}")

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)

    print("Instantiating MultiPVPPet model...")
    multi_pvp_model = MultiPVPPet(base_config=pet_model_cfg, pvp_pattern_ids=args.pattern_ids, no_cuda=args.no_cuda)

    print("-------------------------------------------------\n"
          "BEGIN TRAIN LOOP\n"
          "-------------------------------------------------\n")
    steps, final_loss = multi_pvp_model.train(
        task_train_data=train_data,
        train_batch_size=args.pet_per_gpu_train_batch_size,
        num_train_epochs=args.pet_num_train_epochs,
        max_steps=args.pet_max_steps,
        weight_decay=args.weight_decay,
        models_optimizer_lr=args.learning_rate,
        models_optimizer_eps=args.adam_epsilon,
        models_scheduler_warmup_steps=args.warmup_steps,
        pvp_weights_optimizer_lr=args.mpvp_learning_rate,
        pvp_weights_optimizer_eps=args.mpvp_adam_epsilon,
        pvp_weights_scheduler_warmup_steps=args.mpvp_warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size,
        unlabeled_data=unlabeled_data,
        lm_training=args.lm_training,
        use_logits=False,
        alpha=args.alpha,
        temperature=args.temperature,
    )

    print("-------------------------------------------------\n"
          "BEGIN EVAL LOOP\n"
          "-------------------------------------------------\n")
    final_results = multi_pvp_model.eval(
        metrics=args.metrics,
        eval_data=eval_data,
        priming_data=train_data if args.priming else None,
        batch_size=args.pet_per_gpu_eval_batch_size,
        decoding_strategy=args.decoding_strategy,
    )

    results_path = Path(args.output_dir, "results_0.json")
    while results_path.is_file():
        path_digit = int(results_path.stem[-1])
        results_path = results_path.with_stem(f"results_{path_digit + 1}")
    with open(results_path, 'w') as out:
        json.dump(final_results, out, indent=2, cls=ArrayEncoder)
    print(f"Saved results at {str(results_path)}")


if __name__ == "__main__":
    ### SAMPLE CONFIG ###
    # python main.py
    # --method
    # pet
    # --data_dir
    # /home/alderson/Desktop/MVA/NLP/data/debug/cb/
    # --model_type
    # albert
    # --model_name_or_path
    # albert - base - v2
    # --task_name
    # cb
    # --output_dir
    # /home/alderson/Desktop/MVA/NLP/data/debug/outputs/cb/
    # --pattern_ids
    # 0
    # 1
    # 2
    # --lm_training
    # --unlabeled_examples
    # 32
    # --pet_per_gpu_eval_batch_size
    # 1
    # --no_cuda
    main()
