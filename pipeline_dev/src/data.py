import pandas as pd


class DataPiece():
    def __init__(self, frame: pd.DataFrame, meta = {}) -> None:
        self.frame = frame
        self.meta = meta
