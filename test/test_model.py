import pytest
from model import MyModel
from tensorflow.keras.models import Model


@pytest.fixture
def my_model():
    return MyModel()

@pytest.fixture
def my_dataset():
    my_dataset = model.ds
    return my_dataset

def test_model_is_model_instance(my_model):
    assert isinstance(my_model, Model)

def test_model_with_dataset_for_xxxx(my_model, my_dataset):
    assert my_model.get_layer(name=None, index=0)
