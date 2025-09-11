import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse, hstack, csr_matrix

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ---- FunctionTransformer helpers (pickle-safe; no lambdas) ----
def qmarks_to_nan(X):
    # Convert '?' to NaN so imputer can handle it
    Xdf = pd.DataFrame(X)
    Xdf = Xdf.replace('?', np.nan)
    return Xdf.values

def to_str(X):
    # Force uniform string dtype so OHE doesn't see mixed int/str
    return pd.DataFrame(X).astype(str).values


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = 'artifacts/preprocessor.pkl'


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns=None, categorical_columns=None):
        try:
            logging.info('Data Transformation initiated')

            if numerical_columns is None:
                numerical_columns = [
                    'mb_readmitted_gt30_ct','mb_readmitted_lt30_ct','mb_readmitted_no_ct',
                    'distinct_diag_count','encounter_ct','number_inpatient',
                    'mb_number_inpatient_ct','mb_number_diagnoses_ct'
                ]

            if categorical_columns is None:
                categorical_columns = [
                    'admission_type','A1Cresult'
                ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

            cat_pipeline = Pipeline(steps=[
                ('qmark_to_nan', FunctionTransformer(qmarks_to_nan)),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('to_str', FunctionTransformer(to_str)),
                ('one_hot_encoder', ohe),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys) from e

    def _append_target_sparse_safe(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        if issparse(X):
            return hstack([X, csr_matrix(y)], format='csr')
        return np.c_[X, y]

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'readmitted_ind'
            drop_columns = [target_column_name]

            X_train_df = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column_name]

            X_test_df = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column_name]

            logging.info('Fitting transformer on train, transforming train/test')
            X_train = preprocessing_obj.fit_transform(X_train_df)
            X_test  = preprocessing_obj.transform(X_test_df)

            train_arr = self._append_target_sparse_safe(X_train, y_train)
            test_arr  = self._append_target_sparse_safe(X_test,  y_test)

            logging.info('Saving preprocessing object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys) from e
