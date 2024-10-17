from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple
import pandas as pd
import os
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from openai import OpenAI

class PreprocessConfig(BaseModel):
    rawdata_dir: Path = Field(default = Path("data/raw"), description="direcotry of raw file")
    output_dir: Path = Field(default = Path("data/processed"), description = "directory of processed data")
    training_ratio: float = Field(default = 0.7, description = "training ratio")
    embedding_model: str = Field(default = "text-embedding-3-small", description = "embedding model from openai")
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY"), description="open ai api key")
class Dataset(BaseModel):
    output: pd.DataFrame = Field(..., description = "output dataset ")
    model_config = ConfigDict(arbitrary_types_allowed=True)

def load_data(config: PreprocessConfig) -> Dataset:
    data_dir = Path(config.rawdata_dir)
    for file in data_dir.glob("*.csv"):
        if not file:
            raise ValueError(f"No data on ham vs spam found in the directory? maybe this is a spam?")
        file_name = file.stem
        print("file name", file)
        data = pd.read_csv(file, encoding='ISO-8859-1')
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

class EmbeddingDataset(BaseModel):
    embeddingData: list = Field(..., descriptions="the message converted to embedding")
    # List[List(float)]

def get_embedding(text: str, config: PreprocessConfig):
    #print("api key", config.api_key)
    client = OpenAI(api_key=config.api_key)
    text = text.replace("\n", " ")
    response = client.embeddings.create(input = [text], model=config.embedding_model)
    embedding = response.data[0].embedding
    return embedding


def create_embeddings(config: PreprocessConfig) -> EmbeddingDataset:
    train_data = pd.read_csv(os.path.join(config.output_dir, "train.csv"))
    val_data = pd.read_csv(os.path.join(config.output_dir,"val.csv"))

    
    # train_data_embeddding = [get_embedding(x) for x in train_data[:,1]]
    # val_data_embedding = [get_embedding(x) for x in val_data[:,1]]

    train_data['embedding'] = train_data['v2'].apply(lambda x: get_embedding(x,config))
    val_data['embedding'] = val_data['v2'].apply(lambda x: get_embedding(x,config))

    train_data.to_pickle(os.path.join(config.output_dir, 'train_embeddings.pkl'))
    val_data.to_pickle(os.path.join(config.output_dir, 'val_embeddings.pkl'))

    # for name, data in [("trainEmbedding", train_data_embeddding), ("valEmbedding", val_data_embedding)]:
    #     file_path = os.path.join(config.output_dir, f"{name}.pkl")
    #     data.to_pickle(file_path)

    logger.info("Data conversion to embedding completed")
    return train_data, val_data