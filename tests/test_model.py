"""Tests for src/model.py.

Scope deliberately kept to "does the code do what it claims" (shapes, freeze
behavior) rather than "is the model accurate" — that's what the notebook's
evaluation on the held-out test set is for.
"""

import pytest
import torch

from src.model import TinyVGG, build_resnet


def test_tinyvgg_output_shape():
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=2)
    x = torch.randn(4, 3, 64, 64)

    logits = model(x)

    assert logits.shape == (4, 2)


def test_tinyvgg_hidden_units_changes_param_count():
    small = TinyVGG(input_shape=3, hidden_units=5, output_shape=2)
    large = TinyVGG(input_shape=3, hidden_units=10, output_shape=2)

    small_params = sum(p.numel() for p in small.parameters())
    large_params = sum(p.numel() for p in large.parameters())

    assert large_params > small_params


def test_build_resnet_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_resnet("resnet50", num_classes=2)


def test_build_resnet_frozen_backbone_only_fc_trainable():
    model = build_resnet("resnet18", num_classes=2, freeze_backbone=True)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert trainable == {"fc.weight", "fc.bias"}


def test_build_resnet_unfrozen_backbone_has_more_trainable_params():
    frozen = build_resnet("resnet18", num_classes=2, freeze_backbone=True)
    unfrozen = build_resnet("resnet18", num_classes=2, freeze_backbone=False)

    frozen_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    unfrozen_trainable = sum(p.numel() for p in unfrozen.parameters() if p.requires_grad)

    assert unfrozen_trainable > frozen_trainable


def test_build_resnet_output_shape_matches_num_classes():
    model = build_resnet("resnet18", num_classes=2, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)

    assert logits.shape == (2, 2)
