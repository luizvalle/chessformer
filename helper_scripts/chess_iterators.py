import chess.pgn
import zstandard as zstd
import requests
import io


class CompressedPgnHeaderIterator:
    def __init__(self, download_link):
        dctx = zstd.ZstdDecompressor()
        # Stream the results so we do not load everything
        # into memory at once
        self.response = requests.get(url=download_link, stream=True)
        reader = dctx.stream_reader(self.response.raw)
        self.text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def __iter__(self):
        return self

    def __next__(self):
        header = chess.pgn.read_headers(self.text_stream)
        if header:
            return header
        else:
            self.response.close()
            raise StopIteration

