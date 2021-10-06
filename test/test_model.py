import pytest
from model import MyModel
from tensorflow.keras.models import Model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


@pytest.fixture
def my_model():
    return MyModel()

@pytest.fixture
def my_dataset():
    # return your dataset object
    iris = load_iris()

    # Load data into a DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Convert datatype to float
    df = df.astype(float)
    # append "target" and name it "label"
    df['label'] = iris.target
    # Use string label instead
    df['label'] = df.label.replace(dict(enumerate(iris.target_names)))

    # label -> one-hot encoding
    label = pd.get_dummies(df['label'])
    label.columns = ['label_' + str(x) for x in label.columns]
    df = pd.concat([df, label], axis=1)
    # drop old label
    df.drop(['label'], axis=1, inplace=True)

    # Creating X and y
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    # Convert DataFrame into np array
    X = np.asarray(X)
    y = df[['label_setosa', 'label_versicolor', 'label_virginica']]
    # Convert DataFrame into np array
    y = np.asarray(y)

    # Creating X and y
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    # Convert DataFrame into np array
    X = np.asarray(X)
    y = df[['label_setosa', 'label_versicolor', 'label_virginica']]
    # Convert DataFrame into np array
    y = np.asarray(y)
    
    return X, y

def test_model_is_model_instance(my_model):
    assert isinstance(my_model, Model)

def test_model_with_dataset_for_xxxx(my_model, my_dataset):
    prediction = my_model.predict(x=my_dataset)
    assert(prediction == 'setosa' or prediction == 'versicolor' or prediction == 'virginica')
