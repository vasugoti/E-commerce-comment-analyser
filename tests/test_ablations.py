import pytest
import os
from src.training.run_ablations import ABLATION_REGISTRY, load_ablation_config

def test_ablation_registry_has_all_seven():
    expected_ablations = {
        'label_granularity',
        'preprocessing',
        'embeddings',
        'loss_function',
        'finetuning_depth',
        'sequence_length',
        'data_volume',
    }
    assert set(ABLATION_REGISTRY.keys()) == expected_ablations

@pytest.mark.parametrize("ablation_name", [
    'label_granularity',
    'preprocessing',
    'embeddings',
    'loss_function',
    'finetuning_depth',
    'sequence_length',
    'data_volume',
])
def test_ablation_configs_load_valid_yaml(ablation_name):
    config = load_ablation_config(ablation_name)
    assert isinstance(config, dict)
    assert 'output_dir' in config
