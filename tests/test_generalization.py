import pytest
import pandas as pd
from src.evaluation.generalization import evaluate_zero_shot

def test_evaluate_zero_shot_basic(tmp_path):
    # Dummy train and test dataframes
    source_df = pd.DataFrame({
        'text': ['terrible product broke immediately', 'great quality loved it', 'average item ok', 'bad', 'amazing'],
        'label_5class': [0, 4, 2, 0, 4]
    })
    target_df = pd.DataFrame({
        'text': ['horrible defective', 'fantastic works well'],
        'label_5class': [0, 4]
    })
    
    metrics = evaluate_zero_shot(
        source_df=source_df,
        target_df=target_df,
        dataset_name="TestTarget",
        output_dir=str(tmp_path / "generalization")
    )
    
    assert isinstance(metrics, dict)
    assert 'macro_f1' in metrics
    assert 'accuracy' in metrics
    assert metrics['n_samples'] == 2
