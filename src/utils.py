import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        model_name = getattr(model, "name", model.__class__.__name__)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test,  y_test_pred)
        test_f1   = f1_score(y_test, y_test_pred, average="binary")

        return {"model": model_name, "train_acc": train_acc, "test_acc": test_acc, "test_f1": test_f1}
    except Exception as e:
        raise CustomException(e, sys) from e