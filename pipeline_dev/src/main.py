from pathlib import Path
import yaml
from reader import CsvReader
from components import CleanData, SimpleOut, ChangeIndex,  MultiplyCols, FilterRows, StoreToyData

_IN_FILE_PATH = str(Path(__file__).parents[1] / "input_data.csv")
_OUT_FILE_PATH = str(Path(__file__).parents[1] / "tdb.json")

if __name__ == "__main__":
    csv_reader = CsvReader(sep=";", col_names=["name", "id", "pcs", "factor"])
    outprint = SimpleOut()
    cleaner = CleanData()
    indexer = ChangeIndex("id")
    mul = MultiplyCols("pcs", "factor", "weight")
    filt = FilterRows("pcs", 3)

    in_data = csv_reader.run(_IN_FILE_PATH)
    cleaned_data = cleaner.run(in_data)
    indexed_data = indexer.run(in_data)
    mul_data = mul.run(indexed_data)
    filt_data = filt.run(in_data)
    outprint.run(filt_data)
