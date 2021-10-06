import pytest
from src.model import MyModel
from tensorflow.keras.models import Model

@pytest.fixture
def my_model():
    return MyModel()

# @pytest.fixture
# def my_dataset():
    # return your dataset object

def test_model_is_model_instance(my_model):
    assert isinstance(my_model, Model)

# def test_model_with_dataset_for_xxxx(my_model, my_dataset):
