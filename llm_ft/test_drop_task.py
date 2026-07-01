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
from templates import DROPTemplate


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
        spec = importlib.util.spec_from_file_location("drop_task_module", ROOT / "tasks" / "tasks.py")
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
DROPDataset = task_module.DROPDataset
Sample = task_module.Sample
get_task = task_module.get_task


def _example(passage, question, spans):
    return {
        "passage": passage,
        "question": question,
        "answers_spans": {"spans": spans},
    }


def test_drop_dataset_uses_train_and_validation_splits(monkeypatch):
    calls = []

    def fake_load_dataset(name, *args, **kwargs):
        calls.append(name)
        return {
            "train": [_example("Passage A.", "How many?", ["3"])],
            "validation": [_example("Passage B.", "Who scored?", ["Alice", "Alice Smith"])],
        }

    monkeypatch.setattr(task_module, "load_dataset", fake_load_dataset)

    task = get_task("DROP")

    assert calls == ["drop"]
    assert task.generation is True
    assert task.metric_name == "f1"
    assert len(task.samples["train"]) == 1
    assert len(task.samples["valid"]) == 1

    train_sample = task.samples["train"][0]
    assert train_sample.candidates is None
    assert train_sample.correct_candidate == ["3"]
    assert train_sample.data == {
        "context": "Passage A.",
        "question": "How many?",
        "answers": ["3"],
    }

    valid_sample = task.samples["valid"][0]
    assert valid_sample.correct_candidate == ["Alice", "Alice Smith"]


def test_drop_template_prompt_without_answer_and_answer_target():
    sample = Sample(
        data={"context": "Tom has 2 cats and 1 dog.", "question": "How many pets?", "answers": ["3"]},
        correct_candidate=["3"],
    )
    template = DROPTemplate()

    assert template.encode(sample) == (
        "Passage: Tom has 2 cats and 1 dog.\nQuestion: How many pets?\nAnswer:"
    )
    assert template.verbalize(sample, sample.correct_candidate) == (
        "Passage: Tom has 2 cats and 1 dog.\nQuestion: How many pets?\nAnswer: 3\n"
    )
    with pytest.raises(NotImplementedError):
        template.encode_sfc(sample)
    with pytest.raises(NotImplementedError):
        template.verbalize_sfc(sample, sample.correct_candidate)


def test_drop_f1_metric_exact_partial_and_multi_gold():
    # exact match -> f1 = 1.0
    exact = [types.SimpleNamespace(correct_candidate=["New York"], predicted_candidate="New York")]
    assert calculate_metric(exact, "f1") == 1.0

    # partial token overlap: pred "New" vs gold "New York" -> precision 1.0, recall 0.5
    partial = [types.SimpleNamespace(correct_candidate=["New York"], predicted_candidate="New")]
    assert abs(calculate_metric(partial, "f1") - (2 * 1.0 * 0.5) / (1.0 + 0.5)) < 1e-9

    # multiple gold spans -> metric takes the best (max) f1 over golds
    multi = [types.SimpleNamespace(correct_candidate=["Alice Smith", "Alice"], predicted_candidate="Alice")]
    assert calculate_metric(multi, "f1") == 1.0
