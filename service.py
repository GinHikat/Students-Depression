import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

model_trainer_demo = bentoml.sklearn.get("model:latest").to_runner()

model_api = bentoml.Service('best_model_classifier', runners = [model_trainer_demo])

@model_api.api(input = NumpyNdarray(), output = NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_trainer_demo.predict.run(input_series)
    return result