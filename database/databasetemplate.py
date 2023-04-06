import json
import pandas as pd
import sqlite3
from const.constant import TABLES, ROOT_PATH


class DataBaseTemplate(object):
    path_dict = {"json": "data/original_data/train_data/feature_cluster.json",
                 "train_data": "data/original_data/train_data/train_data.csv",
                 "val_data": "data/original_data/train_data/val_data.csv",
                 "test_data": "data/original_data/test_data/test_data.csv"}

    db_dict = None
    db_root = None

    # db_root_all = ROOT_PATH["all_data"]
    # db_root_features = ROOT_PATH["features_data"]
    # db_root_models = ROOT_PATH["models_data"]

    def __init__(self):
        pass

    @classmethod
    def read_json(cls, file_path: str):
        file = open(file_path, "r")
        content = file.read()
        result = json.loads(content)
        file.close()
        return result

    @classmethod
    def read_csv(cls, file_path: str):
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Unnamed: 0': 'date'})
        return df

    @classmethod
    def get_json(cls):
        file_path = cls.path_dict["json"]
        data = cls.read_json(file_path=file_path)
        return data

    @classmethod
    def get_train_data(cls):
        file_path = cls.path_dict["train_data"]
        data = cls.read_csv(file_path=file_path)
        return data

    @classmethod
    def get_val_data(cls):
        file_path = cls.path_dict["val_data"]
        data = cls.read_csv(file_path=file_path)
        return data

    @classmethod
    def get_test_data(cls):
        file_path = cls.path_dict["test_data"]
        data = cls.read_csv(file_path=file_path)
        return data

    @classmethod
    def _get_db_path(cls, db_name: str):
        return "{}/{}.db".format(cls.db_root, db_name)

    @classmethod
    def _get_data_from_db(cls, db_name: str, table_name: str):
        db_path = cls._get_db_path(db_name=db_name)
        con = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM {}".format(table_name), con=con)
        return df

    @classmethod
    def _store_data_to_db(cls, db_name: str, table_name: str, data: pd.DataFrame):
        db_path = cls._get_db_path(db_name=db_name)
        con = sqlite3.connect(db_path)
        data.to_sql(table_name, con=con, if_exists="replace", index=False)
        pass
