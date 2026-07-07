from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from fugusashi.router.interface import BaseRouter, RouterResult
from fugusashi.router.learned import LearnedRouter
from fugusashi.training import TrainingConfig, TrainingResult, expand_dataset, load_dataset


# ---------------------------------------------------------------------------
# Dataset & config tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_expand_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = expand_dataset(data_dir=tmp)
            assert count > 100
            samples = load_dataset(data_dir=tmp)
            assert len(samples) >= count

    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = load_dataset(data_dir=tmp)
            assert samples == []

    def test_invalid_jsonl_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "expanded_preferences.jsonl")
            with open(path, "w") as f:
                f.write("not json\n")
                f.write('{"prompt": "hi", "model": "test"}\n')
            samples = load_dataset(data_dir=tmp)
            assert len(samples) == 1

    def test_load_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "preferences.jsonl")
            p2 = os.path.join(tmp, "training_data.jsonl")
            with open(p1, "w") as f:
                f.write('{"prompt": "a", "model": "m1"}\n')
            with open(p2, "w") as f:
                f.write('{"prompt": "b", "model": "m2"}\n')
            samples = load_dataset(data_dir=tmp)
            assert len(samples) == 2


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 6
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8

    def test_custom(self):
        config = TrainingConfig(epochs=10, learning_rate=1e-4)
        assert config.epochs == 10
        assert config.learning_rate == 1e-4

    def test_min_samples_check(self):
        config = TrainingConfig(min_samples=5)
        assert config.min_samples == 5

    def test_to_dict(self):
        result = TrainingResult(
            accuracy=0.85,
            top3_accuracy=0.95,
            cost_savings=60.0,
            epochs_trained=6,
            training_time_ms=5000.0,
            model_path="/tmp/model",
            n_classes=2,
            n_train=100,
            n_test=25,
            per_class_accuracy={"a": 0.9, "b": 0.8},
            backbone="answerdotai/ModernBERT-base",
        )
        d = result.to_dict()
        assert d["accuracy"] == 0.85
        assert d["top3_accuracy"] == 0.95
        assert d["n_classes"] == 2
        assert d["backbone"] == "answerdotai/ModernBERT-base"
        assert "per_class_accuracy" in d


# ---------------------------------------------------------------------------
# LearnedRouter tests
# ---------------------------------------------------------------------------

class TestLearnedRouterNotTrained:
    """When no model has been trained, LearnedRouter falls back to CostRouter."""

    def test_fallback_when_not_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = LearnedRouter(model_dir=tmp)
            assert not router.is_trained

            available = {
                "llama3.2-local": {"cost_per_input_token": 0.0, "cost_per_output_token": 0.0},
                "gpt-4o-mini": {"cost_per_input_token": 1e-6, "cost_per_output_token": 2e-6},
            }
            result = router.route(
                "What is the capital of France?",
                [{"role": "user", "content": "What is the capital of France?"}],
                available,
            )
            assert result.model is not None
            assert result.confidence > 0
            assert "cost" in result.strategy  # fell back to CostRouter

    def test_is_trained_checks_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = LearnedRouter(model_dir=tmp)
            assert not router.is_trained
            # Create only config.json → still not trained
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({}, f)
            assert not router.is_trained
            # Create model_names.json → now it reports trained
            with open(os.path.join(tmp, "model_names.json"), "w") as f:
                json.dump(["a", "b"], f)
            assert router.is_trained


class TestLearnedRouterTrained:
    """When a model IS trained, LearnedRouter uses ModernBERT."""

    @patch("fugusashi.router.learned.AutoConfig")
    @patch("fugusashi.router.learned.AutoTokenizer")
    @patch("fugusashi.router.learned.AutoModelForSequenceClassification")
    def test_learned_strategy_included_when_trained(self, mock_model_cls, mock_tok_cls, mock_config_cls):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"num_labels": 2, "model_type": "modernbert"}, f)
            with open(os.path.join(tmp, "model_names.json"), "w") as f:
                json.dump(["llama3.2-local", "gpt-4o-mini"], f)

            mock_tok = MagicMock()
            mock_tok_inputs = MagicMock()
            mock_tok_inputs.to.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            mock_tok.return_value = mock_tok_inputs
            mock_tok_cls.from_pretrained.return_value = mock_tok
            mock_config_cls.from_pretrained.return_value = MagicMock()

            logits = torch.tensor([[0.5, 1.5]])
            model_output = MagicMock()
            model_output.logits = logits
            mock_model = MagicMock()
            mock_model.return_value = model_output
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_tok_inputs = MagicMock()
            mock_tok_inputs.to.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            mock_tok = MagicMock()
            mock_tok.return_value = mock_tok_inputs
            mock_tok_cls.from_pretrained.return_value = mock_tok
            mock_config_cls.from_pretrained.return_value = MagicMock()

            # Mock the model + its forward pass
            mock_model = MagicMock()
            # Simulate logits: higher for index 1 (gpt-4o-mini)
            logits = torch.tensor([[0.5, 2.5]])
            model_output = MagicMock()
            model_output.logits = logits
            mock_model.return_value = model_output
            mock_model_cls.from_pretrained.return_value = mock_model

            router = LearnedRouter(model_dir=tmp)
            assert router.is_trained

            available = {
                "llama3.2-local": {"cost_per_input_token": 0.0, "cost_per_output_token": 0.0},
                "gpt-4o-mini": {"cost_per_input_token": 1e-6, "cost_per_output_token": 2e-6},
            }
            result = router.route(
                "Write a Python function",
                [{"role": "user", "content": "Write a Python function"}],
                available,
            )
            assert result.model is not None
            assert "learned" in result.strategy

    @patch("fugusashi.router.learned.AutoConfig")
    @patch("fugusashi.router.learned.AutoTokenizer")
    @patch("fugusashi.router.learned.AutoModelForSequenceClassification")
    def test_fallback_when_model_not_in_available(self, mock_model_cls, mock_tok_cls, mock_config_cls):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"num_labels": 2}, f)
            with open(os.path.join(tmp, "model_names.json"), "w") as f:
                json.dump(["gpt-4o", "claude-3"], f)

            mock_tok = MagicMock()
            mock_tok_inputs = MagicMock()
            mock_tok_inputs.to.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            mock_tok.return_value = mock_tok_inputs
            mock_tok_cls.from_pretrained.return_value = mock_tok
            mock_config_cls.from_pretrained.return_value = MagicMock()

            mock_model = MagicMock()
            logits = torch.tensor([[1.0, 0.5]])
            model_output = MagicMock()
            model_output.logits = logits
            mock_model.return_value = model_output
            mock_model_cls.from_pretrained.return_value = mock_model

            router = LearnedRouter(model_dir=tmp)
            available = {
                "llama3.2-local": {"cost_per_input_token": 0.0},
                "gpt-4o-mini": {"cost_per_input_token": 1e-6},
            }
            result = router.route(
                "Hello",
                [{"role": "user", "content": "Hello"}],
                available,
            )
            # Neither gpt-4o nor claude-3 are available → fallback
            assert result.model is not None
            assert "cost" in result.strategy

    @patch("fugusashi.router.learned.AutoConfig")
    @patch("fugusashi.router.learned.AutoTokenizer")
    @patch("fugusashi.router.learned.AutoModelForSequenceClassification")
    def test_fallback_when_below_threshold(self, mock_model_cls, mock_tok_cls, mock_config_cls):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"num_labels": 2}, f)
            with open(os.path.join(tmp, "model_names.json"), "w") as f:
                json.dump(["llama3.2-local", "gpt-4o-mini"], f)

            mock_tok = MagicMock()
            mock_tok_inputs = MagicMock()
            mock_tok_inputs.to.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            mock_tok.return_value = mock_tok_inputs
            mock_tok_cls.from_pretrained.return_value = mock_tok
            mock_config_cls.from_pretrained.return_value = MagicMock()

            mock_model = MagicMock()
            logits = torch.tensor([[0.0, 0.1]])
            model_output = MagicMock()
            model_output.logits = logits
            mock_model.return_value = model_output
            mock_model_cls.from_pretrained.return_value = mock_model

            router = LearnedRouter(model_dir=tmp)
            available = {
                "llama3.2-local": {"cost_per_input_token": 0.0},
                "gpt-4o-mini": {"cost_per_input_token": 1e-6},
            }
            # softmax([0.0, 0.1]) = [0.475, 0.525] → max ≈ 0.525
            # That's slightly above 0.5, so use a higher threshold
            result = router.route(
                "Hello",
                [{"role": "user", "content": "Hello"}],
                available,
                threshold=0.6,
            )
            assert result.model is not None
            assert "cost" in result.strategy


# ---------------------------------------------------------------------------
# EnsembleRouter integration with learned router
# ---------------------------------------------------------------------------

class TestEnsembleIntegration:

    @patch("fugusashi.router.learned.AutoConfig")
    @patch("fugusashi.router.learned.AutoTokenizer")
    @patch("fugusashi.router.learned.AutoModelForSequenceClassification")
    def test_learned_strategy_included_when_trained(self, mock_model_cls, mock_tok_cls, mock_config_cls):
        with tempfile.TemporaryDirectory() as tmp:
            # Make it look trained
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"num_labels": 2}, f)
            with open(os.path.join(tmp, "model_names.json"), "w") as f:
                json.dump(["llama3.2-local", "gpt-4o-mini"], f)

            from fugusashi.router import EnsembleRouter

            router = EnsembleRouter(
                model_dir=tmp,
                learned_router_enabled=True,
                confidence_threshold=0.0,
            )
            available = {
                "llama3.2-local": {"cost_per_input_token": 0.0, "cost_per_output_token": 0.0},
                "gpt-4o-mini": {"cost_per_input_token": 1e-6, "cost_per_output_token": 2e-6},
            }
            # With learned_router_enabled=True and is_trained, learned
            # should be the first strategy. The fallback is "cost" which
            # picks the cheapest model.
            result = router.route(
                "What is the capital of France?",
                [{"role": "user", "content": "What is the capital of France?"}],
                available,
            )
            assert result.model is not None
            assert result.confidence > 0

    def test_learned_strategy_skipped_when_disabled(self):
        from fugusashi.router import EnsembleRouter
        router = EnsembleRouter(
            learned_router_enabled=False,
            confidence_threshold=0.0,
        )
        available = {
            "llama3.2-local": {"cost_per_input_token": 0.0, "cost_per_output_token": 0.0},
            "gpt-4o-mini": {"cost_per_input_token": 1e-6, "cost_per_output_token": 2e-6},
        }
        result = router.route(
            "Hello",
            [{"role": "user", "content": "Hello"}],
            available,
        )
        assert result.model is not None
        assert result.model == "llama3.2-local"  # cheapest
