from base import PipelineComponent
from data import DataPiece


class SimpleOut(PipelineComponent):
    def run(self, input: DataPiece):
        print(input.frame)
        print(input.meta)


class CleanData(PipelineComponent):
    def run(self, input: DataPiece):
        input.frame.dropna(inplace=True)
        return input


class ChangeIndex(PipelineComponent):
    def run(self, input: DataPiece):
        input.frame['id'] = input.frame.index.values.tolist()
        return input
