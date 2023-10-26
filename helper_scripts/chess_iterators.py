import chess.pgn
import zstandard as zstd
import requests
import io
import sys


class ResponseRawWrapper(io.IOBase):
    def __init__(self, response_raw):
        self.response_raw = response_raw
        self.num_bytes_read = 0

    def read(self, size=-1):
        data = self.response_raw.read(size)
        self.num_bytes_read += sys.getsizeof(data)
        return data


class CompressedPgnHeaderIterator:
    def __init__(self, download_link):
        dctx = zstd.ZstdDecompressor()
        # Stream the results so we do not load everything
        # into memory at once
        self.response = requests.get(url=download_link, stream=True)
        self.response_raw = ResponseRawWrapper(self.response.raw)
        reader = dctx.stream_reader(self.response_raw)
        self.text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def __iter__(self):
        return self

    def __next__(self):
        header = chess.pgn.read_headers(self.text_stream)
        if header:
            return header
        else:
            raise StopIteration

    def total_num_bytes_read(self):
        return self.response_raw.num_bytes_read

    def total_num_bytes(self):
        return int(self.response.headers.get("Content-Length", 0))
