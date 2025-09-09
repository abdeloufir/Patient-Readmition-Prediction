import os
import sys
import numpy as np
from dataclasses import dataclass

from scipy.sparse import issparse
import xgboost
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # X is all but last col; y is last col. Works for dense or sparse.
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test  = test_array[:, :-1]
            y_test  = test_array[:, -1]

            # ensure 1D dense y
            if issparse(y_train): y_train = y_train.toarray().ravel()
            else:                 y_train = np.asarray(y_train).ravel()

            if issparse(y_test):  y_test  = y_test.toarray().ravel()
            else:                 y_test  = np.asarray(y_test).ravel()

            # (optional) ensure ints for binary labels
            try:
                y_train = y_train.astype(int)
                y_test  = y_test.astype(int)
            except Exception:
                pass

            model = xgboost.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
            )

            model_report = evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                model=model
            )

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, average="binary")
            logging.info(f"Test Accuracy={acc:.4f}, F1={f1:.4f}")

            return {"accuracy": acc, "f1": f1, "report": model_report}

        except Exception as e:
            raise CustomException(e, sys) from e
