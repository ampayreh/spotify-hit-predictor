"""Joblib-serializable wrapper for the Keras MLP model."""

import numpy as np


class KerasModelWrapper:
    """Wraps a Keras .keras model so it can be saved/loaded via joblib."""

    def __init__(self, keras_path, preprocessor):
        self.keras_path = keras_path
        self.preprocessor = preprocessor
        self._model = None

    def _load_model(self):
        if self._model is None:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.keras_path)
        return self._model

    def predict(self, X):
        m = self._load_model()
        X_t = self.preprocessor.transform(X)
        prob = m.predict(X_t, verbose=0).ravel()
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        m = self._load_model()
        X_t = self.preprocessor.transform(X)
        prob = m.predict(X_t, verbose=0).ravel()
        return np.column_stack([1 - prob, prob])
