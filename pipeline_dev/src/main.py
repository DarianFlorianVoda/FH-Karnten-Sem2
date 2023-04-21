from pathlib import Path

from reader import CsvReader
from components import CleanData, SimpleOut, ChangeIndex

_IN_FILE_PATH = str(Path(__file__).parents[1] / "input_data.csv")

if __name__ == "__main__":
    csv_reader = CsvReader(sep=";", col_names=["name", "id", "pcs", "factor"])
    outprint = SimpleOut()
    cleaner = CleanData()
    indexer = ChangeIndex()

    in_data = csv_reader.run(_IN_FILE_PATH)
    cleaned_data = cleaner.run(in_data)
    indexed_data = indexer.run(in_data)
    outprint.run(in_data)
