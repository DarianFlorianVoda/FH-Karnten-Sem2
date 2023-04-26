from base import PipelineComponent
from data import DataPiece
import tinydb as tdb


class SimpleOut(PipelineComponent):
    def run(self, input: DataPiece):
        print(input.frame)
        print(input.meta)


class CleanData(PipelineComponent):
    def run(self, input: DataPiece):
        input.frame.dropna(inplace=True)
        return input


class ChangeIndex(PipelineComponent):
    def __init__(self, col_name) -> None:
        self.name = col_name

    def run(self, input: DataPiece):
        input.frame.set_index(self.name, inplace=True)
        return input


class MultiplyCols(PipelineComponent):
    def __init__(self, col1: str, col2: str, result_name: str) -> None:
        self.col1 = col1
        self.col2 = col2
        self.name = result_name

    def run(self, input: DataPiece):
        input.frame[self.name] = input.frame[self.col1] * input.frame[self.col2]
        return input


class FilterRows(PipelineComponent):
    def __init__(self, cname: str, threshold: float):
        self.threshold = threshold
        self.cname = cname

    def run(self, input: DataPiece):
        input.frame = input.frame.loc[input.frame[self.cname] > self.threshold]
        return input


class StoreToyData(PipelineComponent):
    def __init__(self, path: str):
        self.path = path

    def run(self, input: DataPiece):
        with tdb.TinyDB(self.metadata) as db:
            src_tab = db.table("data_source")
            src_tab.insert(input.meta)

            customer_tab = db.table("customers")
            customer_tab.insert(input.frame.to_dict())
