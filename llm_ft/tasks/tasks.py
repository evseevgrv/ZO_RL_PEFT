import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
from datasets import load_dataset as hf_load_dataset

from templates import *
from utils import temp_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HF_DATASET_CACHE_RETRY_MESSAGES = (
    "must be called with a dataclass type or instance",
    "Feature type 'List' not found",
    "Feature type 'LargeList' not found",
)


def _fallback_hf_cache_dir():
    configured_cache_dir = os.environ.get("ZO_RL_PEFT_HF_DATASETS_CACHE")
    if configured_cache_dir:
        return configured_cache_dir
    return str(Path(tempfile.gettempdir()) / "zo_rl_peft_hf_datasets_cache")


def load_dataset(*args, **kwargs):
    try:
        return hf_load_dataset(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        if not any(message in str(exc) for message in _HF_DATASET_CACHE_RETRY_MESSAGES):
            raise
        if kwargs.get("cache_dir") or kwargs.get("download_mode") == "force_redownload":
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs["cache_dir"] = _fallback_hf_cache_dir()
        retry_kwargs["download_mode"] = "force_redownload"
        logger.warning(
            "Retrying dataset load with a fresh Hugging Face cache at %s after a cache/schema "
            "compatibility error: %s",
            retry_kwargs["cache_dir"],
            exc,
        )
        return hf_load_dataset(*args, **retry_kwargs)


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.samples = None
        self.subtask = subtask

    def get_task_name(self):
        return self.subtask

    def load_dataset(self, path, **kwargs):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def postprocess_generation(self, text):
        return text

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:  # This is always False for now
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed,
                                                            num=num_train + num_dev))  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: SST2Template, 1: SST2TemplateEmpty}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CopaTemplate, 1: CopaTemplateEmpty}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )

        return sample

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: RTETemplate, 1: RTETemplateEmpty}[template_version]()


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class GSM8KDataset(Dataset):
    generation = True

    _number_pattern = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("gsm8k", "main")
        train_examples = dataset["train"]
        valid_examples = dataset["test"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    @classmethod
    def _normalize_number(cls, number):
        number = number.replace(",", "").strip()
        if number.startswith("+"):
            number = number[1:]

        sign = ""
        if number.startswith("-"):
            sign = "-"
            number = number[1:]

        if "." in number:
            whole, fraction = number.split(".", 1)
            whole = whole.lstrip("0") or "0"
            fraction = fraction.rstrip("0")
            if fraction:
                normalized = f"{whole}.{fraction}"
            else:
                normalized = whole
        else:
            normalized = number.lstrip("0") or "0"

        if normalized == "0":
            return "0"
        return sign + normalized

    @classmethod
    def extract_final_answer(cls, text):
        text = str(text).strip()
        if "####" in text:
            text = text.rsplit("####", 1)[-1].strip()

        matches = cls._number_pattern.findall(text)
        if not matches:
            return text
        return cls._normalize_number(matches[-1])

    def build_sample(self, example, idx):
        final_answer = self.extract_final_answer(example["answer"])
        return Sample(
            id=idx,
            data={
                "question": example["question"],
                "answer": example["answer"],
                "final_answer": final_answer,
            },
            candidates=None,
            correct_candidate=final_answer,
        )

    def postprocess_generation(self, text):
        return self.extract_final_answer(text)

    def get_template(self, template_version=0):
        return {0: GSM8KTemplate}[template_version]()


class ARCDataset(Dataset):
    default_subset = "ARC-Challenge"
    subset_aliases = {
        None: "ARC-Challenge",
        "ARC-Challenge": "ARC-Challenge",
        "arc-challenge": "ARC-Challenge",
        "ARCChallenge": "ARC-Challenge",
        "ARC_C": "ARC-Challenge",
        "ARC-C": "ARC-Challenge",
        "challenge": "ARC-Challenge",
        "Challenge": "ARC-Challenge",
        "ARC-Easy": "ARC-Easy",
        "arc-easy": "ARC-Easy",
        "ARCEasy": "ARC-Easy",
        "ARC_E": "ARC-Easy",
        "easy": "ARC-Easy",
        "Easy": "ARC-Easy",
    }

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    @classmethod
    def normalize_subset(cls, subset):
        if subset in cls.subset_aliases:
            return cls.subset_aliases[subset]
        raise ValueError(
            f"Unsupported ARC subset '{subset}'. Use 'ARC-Challenge' or 'ARC-Easy'."
        )

    @staticmethod
    def normalize_choice_label(label):
        return str(label).strip()

    @classmethod
    def resolve_answer_label(cls, answer_key, labels):
        labels = [cls.normalize_choice_label(label) for label in labels]
        answer_key = cls.normalize_choice_label(answer_key)

        if answer_key in labels:
            return answer_key

        upper_to_label = {label.upper(): label for label in labels}
        if answer_key.upper() in upper_to_label:
            return upper_to_label[answer_key.upper()]

        if answer_key.isdigit():
            index = int(answer_key) - 1
            if 0 <= index < len(labels):
                return labels[index]

        if len(answer_key) == 1 and answer_key.isalpha():
            index = ord(answer_key.upper()) - ord("A")
            if 0 <= index < len(labels):
                return labels[index]

        raise ValueError(f"Could not resolve ARC answer key '{answer_key}' from labels {labels}")

    def load_dataset(self, path, **kwargs):
        subset = self.normalize_subset(path)
        dataset = load_dataset("allenai/ai2_arc", subset)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
        self.subtask = subset

    def build_sample(self, example):
        labels = [self.normalize_choice_label(label) for label in example["choices"]["label"]]
        texts = [str(text).strip() for text in example["choices"]["text"]]
        correct_label = self.resolve_answer_label(example["answerKey"], labels)

        return Sample(
            id=example.get("id"),
            data={
                "id": example.get("id"),
                "question": example["question"],
                "choices": {
                    "label": labels,
                    "text": texts,
                },
                "answerKey": correct_label,
            },
            candidates=labels,
            correct_candidate=correct_label,
        )

    def get_template(self, template_version=0):
        return {0: ARCTemplate}[template_version]()


class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_set = load_dataset('winogrande', 'winogrande_m', split='train')
        valid_set = load_dataset('winogrande', 'winogrande_m', split='validation')

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")
