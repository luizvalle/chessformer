import pandas as pd
import re
import multiprocess
import requests

from chess_iterators import CompressedPgnHeaderIterator
from gdrive import GDrive
from multiprocess import Pool
from tqdm import tqdm
from zstd import ZstdError


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1Cwnlq0ziqLP6h0LsZlrzs3lLTHGRDPLI"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
CHUNK_SIZE = 5
N_PROCESSES = multiprocess.cpu_count() + 2


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet.zstd"
    return path


def process_headers(download_link):
    column_types = {
            "Event": "category",
            "Result": "category",
            "WhiteElo": "uint16",
            "BlackElo": "uint16",
            "TimeControl": "category",
            "Termination": "category"
            }
    new_file_name = file_name_from_link(download_link)
    headers = CompressedPgnHeaderIterator(download_link)
    gDrive = GDrive(CREDENTIALS_JSON)
    try:
        for header in headers:
            pass
    except ZstdError as e:
        print(e)
    #_info = [{
    #Event": header["Event"],
    #Result": header["Result"],
    #WhiteElo": int(header["WhiteElo"]) if header["WhiteElo"].isnumeric() else 0,
    #BlackElo": int(header["BlackElo"]) if header["BlackElo"].isnumeric() else 0,
    #TimeControl": header["TimeControl"],
    #Termination": header["Termination"]}
    #or header in headers]
    # df = pd.DataFrame(data=games_info).astype(column_types)
    # gDrive.write_dataframe(df, PARENT_DIR_ID, new_file_name)


def main():
    process_headers("https://database.lichess.org/standard/lichess_db_standard_rated_2017-06.pgn.zst")


if __name__ == "__main__":
    main()

