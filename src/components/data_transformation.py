import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = 'artifacts/preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns=None, categorical_columns=None):
        try:
            logging.info('Data Transformation initiated')
            # if caller did not provide explicit lists, use the original defaults
            if numerical_columns is None:
                numerical_columns = ['time_in_hospital', 'num_lab_procedures', 
                                 'num_procedures', 'num_medications', 'number_outpatient', 
                                 'number_emergency', 'number_inpatient', 'number_diagnoses', 
                                 'MP_DM_payer_ind', 'mb_admission_grp_1_ct', 'mb_admission_grp_2_ct', 
                                 'mb_discharge_grp_1_ct', 'mb_discharge_grp_2_ct', 'mb_admission_type_ct', 
                                 'distinct_diag_count', 'diag_1_freq', 'diag_2_freq', 'diag_3_freq', 
                                 'diag_1_428_ind', 'diag_1_491_ind', 'diag_1_493_ind', 'diag_2_403_ind', 
                                 'diag_2_707_ind', 'diag_2_585_ind', 'diag_2_491_ind', 'diag_3_403_ind', 
                                 'diag_3_585_ind', 'diag_3_707_ind', 'diag_1_driver_ind', 
                                 'diag_2_driver_ind', 'diag_3_driver_ind', 'diagnosis_tuple_freq', 
                                 'diag_1_sort_freq', 'diag_2_sort_freq', 'diag_3_sort_freq', 
                                 'high propensity tuple ind', 'dx_428_ind_max', 'dx_428_ind_sum', 
                                 'dx_403_ind_max', 'dx_403_ind_sum', 'dx_707_ind_max', 'dx_707_ind_sum', 
                                 'dx_585_ind_max', 'dx_585_ind_sum', 'dx_491_ind_max', 'dx_491_ind_sum', 
                                 'dx_396_ind_max', 'dx_396_ind_sum', 'dx_440_ind_max', 'dx_440_ind_sum', 
                                 'dx_453_ind_max', 'dx_453_ind_sum', 'dx_571_ind_max', 'dx_571_ind_sum', 
                                 'dx_284_ind_max', 'dx_284_ind_sum', 'dx_304_ind_max', 'dx_304_ind_sum', 
                                 'dx_482_ind_max', 'dx_482_ind_sum', 'dx_150_ind_max', 'dx_150_ind_sum', 
                                 'dx_282_ind_max', 'dx_282_ind_sum', 'dx_332_ind_max', 'dx_332_ind_sum', 
                                 'dx_443_ind_max', 'dx_443_ind_sum', 'dx_719_ind_max', 'dx_719_ind_sum', 
                                 'dx_423_ind_max', 'dx_423_ind_sum', 'dx_281_ind_max', 'dx_281_ind_sum', 
                                 'dx_536_ind_max', 'dx_536_ind_sum', 'dx_368_ind_max', 'dx_368_ind_sum', 
                                 'dx_515_ind_max', 'dx_515_ind_sum', 'dx_595_ind_max', 'dx_595_ind_sum', 
                                 'dx_572_ind_max', 'dx_572_ind_sum', 'dx_681_ind_max', 'dx_681_ind_sum', 
                                 'dx_581_ind_max', 'dx_581_ind_sum', 'dx_537_ind_max', 'dx_537_ind_sum', 
                                 'dx_490_ind_max', 'dx_490_ind_sum', 'dx_583_ind_max', 'dx_583_ind_sum', 
                                 'dx_V46_ind_max', 'dx_V46_ind_sum', 'dx_519_ind_max', 'dx_519_ind_sum', 
                                 'dx_300_ind_max', 'dx_300_ind_sum', 'dx_567_ind_max', 'dx_567_ind_sum', 
                                 'dx_E92_ind_max', 'dx_E92_ind_sum', 'dx_V49_ind_max', 'dx_V49_ind_sum', 
                                 'dx_094_ind_max', 'dx_094_ind_sum', 'dx_514_ind_max', 'dx_514_ind_sum', 
                                 'dx_494_ind_max', 'dx_494_ind_sum', 'dx_042_ind_max', 'dx_042_ind_sum', 
                                 'dx_404_ind_max', 'dx_404_ind_sum', 'dx_346_ind_max', 'dx_346_ind_sum', 
                                 'dx_792_ind_max', 'dx_792_ind_sum', 'dx_398_ind_max', 'dx_398_ind_sum', 
                                 'dx_753_ind_max', 'dx_753_ind_sum', 'dx_577_ind_max', 'dx_577_ind_sum', 
                                 'dx_730_ind_max', 'dx_730_ind_sum', 'dx_444_ind_max', 'dx_444_ind_sum', 
                                 'dx_459_ind_max', 'dx_459_ind_sum', 'dx_790_ind_max', 'dx_790_ind_sum', 
                                 'dx_337_ind_max', 'dx_337_ind_sum', 'dx_397_ind_max', 'dx_397_ind_sum', 
                                 'dx_292_ind_max', 'dx_292_ind_sum', 'dx_V42_ind_max', 'dx_V42_ind_sum', 
                                 'dx_289_ind_max', 'dx_289_ind_sum', 'alcohol_history_ind', 
                                 'obesity_history_ind', 'mh_history_ind', 
                                 'encounter_ct', 'mb_time_in_hospital', 'mb_readmitted_lt30_ct', 
                                 'mb_readmitted_gt30_ct', 'mb_readmitted_no_ct', 'mb_num_lab_procedures_ct', 
                                 'mb_num_procedures_ct', 'mb_num_medications_ct', 'mb_number_outpatient_ct', 
                                 'mb_number_emergency_ct', 'mb_number_inpatient_ct', 'mb_number_diagnoses_ct', 
                                 'mb A1c gt 7 ct', 'mb A1c gt 8 ct', 'mb A1c Norm ct']
            if categorical_columns is None:
                categorical_columns = ['race', 'gender', 'age', 'weight', 'medical_specialty', 
                                   'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 
                                   'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 
                                   'rosiglitazone', 'insulin', 'change', 'diabetesMed', 
                                   'admission_type', 'discharge_disposition', 'admission_source', 
                                   'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diag_1_sort', 
                                   'diag_2_sort', 'diag_3_sort', 'diagnosis_1_sort', 
                                   'diagnosis_2_sort', 'diagnosis_3_sort']

            num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info('Numerical columns standard scaling completed')
            logging.info('Categorical columns encoding completed')
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'readmitted_ind'
            drop_columns = ['readmitted_ind']
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info('Applied preprocessing object on training and testing datasets')
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info('Saved preprocessing object')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys) from e