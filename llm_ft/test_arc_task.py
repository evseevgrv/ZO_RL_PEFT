import contextlib
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from templates import ARCTemplate


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
        spec = importlib.util.spec_from_file_location("arc_task_module", ROOT / "tasks" / "tasks.py")
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
ARCDataset = task_module.ARCDataset
Sample = task_module.Sample
get_task = task_module.get_task


def test_arc_dataset_uses_arc_challenge_train_and_validation_splits(monkeypatch):
    calls = []

    def fake_load_dataset(name, config):
        calls.append((name, config))
        return {
            "train": [
                {
                    "id": "train-1",
                    "question": "Which planet is known as the Red Planet?",
                    "choices": {
                        "label": ["A", "B", "C", "D"],
                        "text": ["Earth", "Mars", "Jupiter", "Venus"],
                    },
                    "answerKey": "B",
                }
            ],
            "validation": [
                {
                    "id": "valid-1",
                    "question": "Water freezes at what temperature in Celsius?",
                    "choices": {
                        "label": ["1", "2", "3", "4"],
                        "text": ["100", "50", "0", "-10"],
                    },
                    "answerKey": "3",
                }
            ],
        }

    monkeypatch.setattr(task_module, "load_dataset", fake_load_dataset)

    task = get_task("ARC")

    assert calls == [("allenai/ai2_arc", "ARC-Challenge")]
    assert task.subtask == "ARC-Challenge"
    assert len(task.samples["train"]) == 1
    assert len(task.samples["valid"]) == 1

    train_sample = task.samples["train"][0]
    valid_sample = task.samples["valid"][0]
    assert train_sample.correct_candidate == "B"
    assert train_sample.candidates == ["A", "B", "C", "D"]
    assert valid_sample.correct_candidate == "3"
    assert valid_sample.candidates == ["1", "2", "3", "4"]


def test_arc_dataset_accepts_explicit_arc_easy_subset(monkeypatch):
    calls = []

    def fake_load_dataset(name, config):
        calls.append((name, config))
        return {"train": [], "validation": []}

    monkeypatch.setattr(task_module, "load_dataset", fake_load_dataset)

    task = get_task("ARC__ARC-Easy")

    assert calls == [("allenai/ai2_arc", "ARC-Easy")]
    assert task.subtask == "ARC-Easy"


def test_arc_answer_key_resolution_supports_numeric_and_letter_aliases():
    assert ARCDataset.resolve_answer_label("2", ["A", "B", "C", "D"]) == "B"
    assert ARCDataset.resolve_answer_label("c", ["A", "B", "C", "D"]) == "C"
    assert ARCDataset.resolve_answer_label("D", ["1", "2", "3", "4"]) == "4"


def test_load_dataset_retries_with_fresh_cache_on_incompatible_metadata(monkeypatch):
    calls = []

    def fake_hf_load_dataset(name, config, **kwargs):
        calls.append((name, config, kwargs))
        if len(calls) == 1:
            raise TypeError("must be called with a dataclass type or instance")
        return {"train": [], "validation": []}

    monkeypatch.setattr(task_module, "hf_load_dataset", fake_hf_load_dataset)

    task = get_task("ARC")

    assert task.samples == {"train": [], "valid": []}
    assert len(calls) == 2
    assert calls[0] == ("allenai/ai2_arc", "ARC-Challenge", {})
    assert calls[1][0] == "allenai/ai2_arc"
    assert calls[1][1] == "ARC-Challenge"
    assert calls[1][2]["download_mode"] == "force_redownload"
    assert "cache_dir" in calls[1][2]


def test_arc_template_formats_question_choices_and_answer_label():
    sample = Sample(
        data={
            "question": "Which gas do plants absorb from the atmosphere?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
            },
            "answerKey": "B",
        },
        candidates=["A", "B", "C", "D"],
        correct_candidate="B",
    )
    template = ARCTemplate()

    assert template.encode(sample) == (
        "Question: Which gas do plants absorb from the atmosphere?\n"
        "Choices:\n"
        "A. Oxygen\n"
        "B. Carbon dioxide\n"
        "C. Nitrogen\n"
        "D. Helium\n"
        "Answer:"
    )
    assert template.verbalize(sample, sample.correct_candidate) == (
        "Question: Which gas do plants absorb from the atmosphere?\n"
        "Choices:\n"
        "A. Oxygen\n"
        "B. Carbon dioxide\n"
        "C. Nitrogen\n"
        "D. Helium\n"
        "Answer: B"
    )
