import pytest
import pandas as pd
from src.data.ingest import validate_text_column, validate_rating_column

def test_validate_text_column_content_length():
    """
    Test that validate_text_column selects the column with the longer median
    string length, avoiding the failure mode where a short 'title' or 'summary'
    field is picked instead of the actual review body.
    """
    df = pd.DataFrame({
        'short_summary': ['Good', 'Bad', 'Okay', 'Nice', 'Terrible'],
        'actual_review': [
            'This is a much longer review that actually explains why the product is good. I really enjoyed it.',
            'This was a terrible experience. The item broke immediately and the seller was unhelpful.',
            'It is okay, nothing special but it gets the job done for the price point.',
            'Nice product, fast shipping, works as described in the listing.',
            'Do not buy this! Total waste of money and time. Very disappointed.'
        ],
        'unrelated_id': [123, 124, 125, 126, 127]
    })
    
    candidates = ['short_summary', 'actual_review', 'unrelated_id']
    selected = validate_text_column(df, candidates, min_median_length=10)
    
    assert selected == 'actual_review'

def test_validate_text_column_no_valid_column():
    """
    Test that a ValueError is raised when no column meets the criteria.
    """
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [1.1, 2.2, 3.3]
    })
    
    with pytest.raises(ValueError):
         validate_text_column(df, ['col1', 'col2'])

def test_validate_rating_column_valid():
    """
    Test that validate_rating_column correctly identifies a column with
    values between 1 and 5.
    """
    df = pd.DataFrame({
        'ids': [100, 101, 102, 103, 104],
        'scores': [1, 5, 3, 4, 2],
        'prices': [19.99, 5.50, 100.0, 45.0, 99.99]
    })
    
    candidates = ['ids', 'scores', 'prices']
    selected = validate_rating_column(df, candidates)
    
    assert selected == 'scores'

def test_validate_rating_column_string_numbers():
    """
    Test that validate_rating_column can handle string representations of numbers.
    """
    df = pd.DataFrame({
        'rating_str': ['1', '5', '3', '4', '2'],
        'text': ['a', 'b', 'c', 'd', 'e']
    })
    
    candidates = ['rating_str', 'text']
    selected = validate_rating_column(df, candidates)
    
    assert selected == 'rating_str'

def test_validate_rating_column_no_valid():
    """
    Test that a ValueError is raised when no valid rating column exists.
    """
    df = pd.DataFrame({
        'large_nums': [10, 20, 30, 40, 50],
        'text': ['a', 'b', 'c', 'd', 'e']
    })
    
    candidates = ['large_nums', 'text']
    
    with pytest.raises(ValueError):
        validate_rating_column(df, candidates)
