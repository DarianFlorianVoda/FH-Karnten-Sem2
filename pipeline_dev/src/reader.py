import pandas as pd

from base import PipelineComponent
from data import DataPiece


class CsvReader(PipelineComponent):
    def __init__(self, sep: str, col_names: list):
        self.sep = sep
        self.names = col_names

    def run(self, path: str):
        csv_data = pd.read_csv(filepath_or_buffer=path, sep=self.sep, names=self.names)
        meta = {"file_path": path}
        return DataPiece(csv_data, meta)
