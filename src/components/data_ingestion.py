import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = artifacts_dir / "train.csv"
    test_data_path: Path  = artifacts_dir / "test.csv"
    raw_data_path: Path   = artifacts_dir / "data.csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _ensure_artifacts_dir(self):
        if self.ingestion_config.artifacts_dir.exists() and not self.ingestion_config.artifacts_dir.is_dir():
            raise CustomException(
                f"'artifacts' exists but is not a directory: {self.ingestion_config.artifacts_dir}", sys
            )
        self.ingestion_config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion")
        try:
            self._ensure_artifacts_dir()

            src_csv = Path("notebook") / "data" / "p004_diabetes_MLprep.csv"
            logging.info(f"Reading dataset from: {src_csv.resolve()}")
            df = pd.read_csv(src_csv)

            logging.info(f"Writing raw data to: {self.ingestion_config.raw_data_path.resolve()}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train/test split (test_size=0.20, random_state=42)")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            logging.info(f"Writing train to: {self.ingestion_config.train_data_path.resolve()}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)

            logging.info(f"Writing test to: {self.ingestion_config.test_data_path.resolve()}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion complete")
            return (str(self.ingestion_config.train_data_path),
                    str(self.ingestion_config.test_data_path))

        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
