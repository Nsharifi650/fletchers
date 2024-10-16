from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import os
import numpy as np
from pathlib import Path
from src.utils.logger import logger

class preprocessConfig(BaseModel):
    rawdata_dir: Path = Field(default = Path("data/raw"), description="direcotry of raw file")
    output_dir: Path = Field(default = Path("data/processed"), description = "directory of processed data")
    training_ratio: float = Field(default = 0.7, description = "training ratio")

class Dataset(BaseModel):
    output: pd.DataFrame = Field(..., description = "output dataset ")
    model_config = ConfigDict(arbitrary_types_allowed=True)

def load_data(config: preprocessConfig) -> Dataset:
    data_dir = Path(config.rawdata_dir)
    for file in data_dir.glob("*.csv"):
        if not file:
            raise ValueError(f"No data on ham found in the directory? maybe this is a spam?")
        file_name = file.stem
        data = pd.read_csv(file)
        logger.info(f"Finished Uploading local data {file_name}")
    return Dataset(output=data)


def split_data(
        dataset: Dataset, 
        config: PreprocessConfig
        ) -> Tuple[Dataset, Dataset]:
    dataset_length = len(dataset.output)
    training_len = int(config.training_ratio*dataset_length)

    training = dataset.output.iloc[:training_len,:]
    validation = dataset.output.iloc[training_len:,:]
    return (
        Dataset(output= training),
        Dataset(output= validation)
    )

def preprocess_data(
        config:PreprocessConfig
        ) -> Tuple[Dataset,Dataset]:

    data = load_data(config)

    # split the data into trianing, validation and testing sets
    (train_data, val_data) = split_data(data,config)

    # save this processed data into output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok =True)
    

    for name, data in [("train", train_data), ("val", val_data)]:
        file_path = os.path.join(output_dir, f"{name}.csv")
        data.output.to_csv(file_path)
            
    logger.info("Data preprocessing completed")
    return train_data, val_data

# conversion of words to embedding

