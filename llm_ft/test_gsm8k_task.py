import sys
import types
import importlib.util
import contextlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import calculate_metric
from templates import GSM8KTemplate


@contextlib.contextmanager
def temp_seed(seed):
    yield


def load_task_module():
    previous_utils = sys.modules.get("utils")
    previous_datasets = sys.modules.get("datasets")
    sys.modules["utils"] = types.SimpleNamespace(temp_seed=temp_seed)
    sys.modules["datasets"] = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("load_dataset must be monkeypatched in tests")
        )
    )
    try:
        spec = importlib.util.spec_from_file_location("gsm8k_task_module", ROOT / "tasks" / "tasks.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
        if previous_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = previous_datasets


task_module = load_task_module()
GSM8KDataset = task_module.GSM8KDataset
Sample = task_module.Sample
get_task = task_module.get_task


def test_gsm8k_dataset_uses_train_and_test_splits(monkeypatch):
    calls = []

    def fake_load_dataset(name, config):
        calls.append((name, config))
        return {
            "train": [
                {
                    "question": "How many apples are there?",
                    "answer": "There are 1,000 plus 234 apples. #### 1,234",
                }
            ],
            "test": [
                {
                    "question": "What is 40 plus 2?",
                    "answer": "40 + 2 = 42. #### 42",
                }
            ],
        }

    monkeypatch.setattr(task_module, "load_dataset", fake_load_dataset)

    task = get_task("GSM8K")

    assert calls == [("gsm8k", "main")]
    assert len(task.samples["train"]) == 1
    assert len(task.samples["valid"]) == 1
    assert task.generation is True

    train_sample = task.samples["train"][0]
    valid_sample = task.samples["valid"][0]
    assert train_sample.correct_candidate == "1234"
    assert train_sample.data == {
        "question": "How many apples are there?",
        "answer": "There are 1,000 plus 234 apples. #### 1,234",
        "final_answer": "1234",
    }
    assert valid_sample.correct_candidate == "42"


def test_gsm8k_template_uses_prompt_without_answer_and_full_solution_target():
    sample = Sample(
        data={
            "question": "What is 40 plus 2?",
            "answer": "40 + 2 = 42. #### 42",
            "final_answer": "42",
        },
        correct_candidate="42",
    )
    template = GSM8KTemplate()

    assert template.encode(sample) == "Question: What is 40 plus 2?\nAnswer:"
    assert template.verbalize(sample, sample.correct_candidate) == (
        "Question: What is 40 plus 2?\nAnswer: 40 + 2 = 42. #### 42\n"
    )
    with pytest.raises(NotImplementedError):
        template.encode_sfc(sample)
    with pytest.raises(NotImplementedError):
        template.verbalize_sfc(sample, sample.correct_candidate)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning #### 42", "42"),
        ("Reasoning gives 1,234 apples", "1234"),
        ("#### -3.50", "-3.5"),
        ("It is 42.0", "42"),
        ("No numeric answer", "No numeric answer"),
    ],
)
def test_gsm8k_postprocess_generation_extracts_final_answer(text, expected):
    dataset = object.__new__(GSM8KDataset)

    assert dataset.postprocess_generation(text) == expected


def test_gsm8k_accuracy_uses_existing_accuracy_metric():
    predictions = [types.SimpleNamespace(correct_candidate="42", predicted_candidate="42")]

    assert calculate_metric(predictions, "accuracy") == 1.0
