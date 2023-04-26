from pathlib import Path
import yaml
from reader import CsvReader
from components import CleanData, SimpleOut, ChangeIndex,  MultiplyCols, FilterRows, StoreToyData

_IN_FILE_PATH = str(Path(__file__).parents[1] / "input_data.csv")
_OUT_FILE_PATH = str(Path(__file__).parents[1] / "tdb.json")


if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    csv_reader = CsvReader(**config['CsvReader'])
    outprint = SimpleOut()
    cleaner = CleanData()
    indexer = ChangeIndex(**config['ChangeIndex'])
    mul = MultiplyCols(**config['MultiplyCols'])
    filt = FilterRows(**config['FilterRows'])
    writedb = StoreToyData(**config['StoreToyData'])

    in_data = csv_reader.run(_IN_FILE_PATH)
    cleaned_data = cleaner.run(in_data)
    indexed_data = indexer.run(in_data)
    mul_data = mul.run(indexed_data)
    filt_data = filt.run(in_data)
    outprint.run(filt_data)
