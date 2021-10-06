import pytest
from model import MyModel
from tensorflow.keras.models import Model


@pytest.fixture
def my_model():
    return MyModel()

@pytest.fixture
# def my_dataset():
    # return your dataset object

def test_model_is_model_instance(my_model: MyModel):
    assert isinstance(my_model, Model)

def test_model_returns_same_type(my_model: MyModel):
    assert my_model.call([1,2,3,4]) is not None

# def test_model_with_dataset_for_xxxx(my_model, my_dataset):
