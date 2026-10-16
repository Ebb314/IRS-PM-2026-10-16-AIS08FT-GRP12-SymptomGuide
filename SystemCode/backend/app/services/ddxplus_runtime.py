"""Load the teammates' tri-state recall models.

  load_ddxplus_model()   app/recall_model/       DDXPlus, 49 diseases x 95 evidences
  load_recall773_model() app/recall_model_773/   original dataset, 773 diseases x 377 symptoms

Both use the same Two-Tower network (teammate class disease_retrieval.TwoTower):
input = N "present" + N "absent" channels (unknown = both 0); patient tower 2N -> 128 -> 64;
score = patient vector . disease embedding / 8 (uncalibrated logits).

Backends (DDXPLUS_BACKEND / RECALL773_BACKEND = auto | torch | numpy):
  torch  rebuilds that network with PyTorch and loads the teammate's weights file.
         (The teammate modules are not imported: both bundles ship a disease_retrieval.py whose
         ROOT is fixed at import time, so importing both in one process would mix their paths.)
  numpy  the same forward pass in numpy, reading the same weights file.
  auto   torch when importable, otherwise numpy. Equivalence is checked by the tests
         (example_predictions.json, and torch == numpy when torch is installed).
"""
import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
PACKAGE = APP / "recall_model"
ARTIFACTS = PACKAGE / "artifacts" / "disease_retrieval_tristate"
PACKAGE_773 = APP / "recall_model_773"
ARTIFACTS_773 = PACKAGE_773 / "artifacts" / "disease_retrieval_773_tristate_v2"
KEYS = ("patient.0.weight", "patient.0.bias", "patient.2.weight", "patient.2.bias", "disease.weight")


def _load_state_dict_numpy(path):
    """Read a PyTorch zip state_dict of float32 tensors without importing torch."""
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_):
        base = storage[offset:]
        return np.lib.stride_tricks.as_strided(
            base, shape=size, strides=[s * base.itemsize for s in stride]).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "OrderedDict":
                return OrderedDict
            if module.startswith("torch") and name == "FloatStorage":
                return name
            raise pickle.UnpicklingError(f"Unexpected object in weights file: {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, *_ = pid
            if storage_type != "FloatStorage":
                raise pickle.UnpicklingError(f"Unsupported storage {storage_type}")
            return np.frombuffer(archive.read(f"{prefix}/data/{key}"), dtype="<f4")

    return Unpickler(archive.open(f"{prefix}/data.pkl")).load()


class _NumpyTwoTower:
    def __init__(self, weights_path):
        state = _load_state_dict_numpy(weights_path)
        self.w0, self.b0, self.w1, self.b1, self.diseases = (state[k] for k in KEYS)

    def logits(self, vector):
        hidden = np.maximum(self.w0 @ vector + self.b0, 0.0)
        return self.diseases @ (self.w1 @ hidden + self.b1) / 8


class _TorchTwoTower:
    """Same layers as the teammates' disease_retrieval.TwoTower."""

    def __init__(self, weights_path, inputs, diseases):
        import torch
        from torch import nn
        self.torch = torch
        self.patient = nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 64))
        self.disease = nn.Embedding(diseases, 64)
        module = nn.Module()
        module.patient, module.disease = self.patient, self.disease
        module.load_state_dict(torch.load(weights_path, weights_only=True))
        module.eval()

    def logits(self, vector):
        with self.torch.no_grad():
            x = self.torch.from_numpy(vector[None])
            return (self.patient(x) @ self.disease.weight.T / 8).numpy()[0]


class TristateRecallModel:
    """features: model symptom IDs/names; diseases: label list; logits(present, absent)."""

    def __init__(self, backend="auto", weights_path=ARTIFACTS / "tristate.pt", features=None, diseases=None):
        if features is None or diseases is None:
            vocab = json.loads((ARTIFACTS / "vocabulary.json").read_text(encoding="utf-8"))
            features, diseases = vocab["evidence_ids"], vocab["diseases"]
        self.features, self.diseases = list(features), list(diseases)
        self.evidence_ids = self.features  # DDXPlus naming, kept for compatibility
        self.index = {f: i for i, f in enumerate(self.features)}
        self.backend, self._net = None, None
        if backend in ("auto", "torch"):
            try:
                self._net = _TorchTwoTower(weights_path, 2 * len(self.features), len(self.diseases))
                self.backend = "torch"
            except ImportError:
                if backend == "torch":
                    raise
        if self._net is None:
            self._net = _NumpyTwoTower(weights_path)
            self.backend = "numpy"

    def encode(self, present, absent):
        present, absent = set(present), set(absent)
        unknown = (present | absent) - set(self.index)
        if unknown:
            raise ValueError(f"Symptoms outside the model vocabulary: {sorted(unknown)}")
        if present & absent:
            raise ValueError(f"Contradictory evidence: {sorted(present & absent)}")
        vector = np.zeros(2 * len(self.features), dtype=np.float32)
        for f in present:
            vector[self.index[f]] = 1
        for f in absent:
            vector[len(self.features) + self.index[f]] = 1
        return vector

    def logits(self, present, absent=()):
        return self._net.logits(self.encode(present, absent))

    def probabilities(self, present, absent=(), temperature=1.0):
        z = self.logits(present, absent) / temperature
        z = np.exp(z - z.max())
        return z / z.sum()


def load_ddxplus_model(backend="auto"):
    return TristateRecallModel(backend)


def load_recall773_model(backend="auto", variant=None):
    vocab = json.loads((ARTIFACTS_773 / "vocabulary.json").read_text(encoding="utf-8"))
    variant = variant or json.loads((ARTIFACTS_773 / "selection.json").read_text(encoding="utf-8"))["selected_model"]
    if variant not in ("ce", "ce_pairwise"):
        raise ValueError(f"Unknown 773 model variant: {variant}")
    model = TristateRecallModel(backend, ARTIFACTS_773 / f"{variant}.pt", vocab["symptoms"], vocab["diseases"])
    model.variant = variant
    return model
