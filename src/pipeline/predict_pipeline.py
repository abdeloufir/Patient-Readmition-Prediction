import sys
import pandas as pd
from typing import Optional
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys) from e
        
class CustomData:
    def __init__(self,
                 mb_readmitted_gt30_ct: int,
                 mb_readmitted_lt30_ct: int,
                 mb_readmitted_no_ct: int,
                 distinct_diag_count: int,    
                 encounter_ct: int,    
                 number_inpatient: int,    
                 mb_number_inpatient_ct: int,    
                 mb_number_diagnoses_ct: int,    
                 admission_type: Optional[str] = None,    
                 A1Cresult: Optional[str] = None,
                ):
        self.mb_readmitted_gt30_ct = mb_readmitted_gt30_ct
        self.mb_readmitted_lt30_ct = mb_readmitted_lt30_ct
        self.mb_readmitted_no_ct = mb_readmitted_no_ct
        self.distinct_diag_count = distinct_diag_count
        self.encounter_ct = encounter_ct
        self.number_inpatient = number_inpatient
        self.mb_number_inpatient_ct = mb_number_inpatient_ct
        self.mb_number_diagnoses_ct = mb_number_diagnoses_ct
        self.admission_type = admission_type
        self.A1Cresult = A1Cresult
                 
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "mb_readmitted_gt30_ct": [self.mb_readmitted_gt30_ct],
                "mb_readmitted_lt30_ct": [self.mb_readmitted_lt30_ct],
                "mb_readmitted_no_ct": [self.mb_readmitted_no_ct],
                "distinct_diag_count": [self.distinct_diag_count],
                "encounter_ct": [self.encounter_ct],
                "number_inpatient": [self.number_inpatient],
                "mb_number_inpatient_ct": [self.mb_number_inpatient_ct],
                "mb_number_diagnoses_ct": [self.mb_number_diagnoses_ct],
                "admission_type": [self.admission_type],
                "A1Cresult": [self.A1Cresult],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e