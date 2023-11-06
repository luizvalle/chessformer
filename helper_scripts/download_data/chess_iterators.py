import chess
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


class Game:
    def __init__(self):
        self.headers = dict()
        self.moves = list()
        self.had_parsing_errors = False


class FastGameVisitor(chess.pgn.BaseVisitor):
    def begin_game(self):
        self.game = Game()
    
    def visit_header(self, tagname: str, tagvalue: str):
        self.game.headers[tagname] = tagvalue

    def end_headers(self):
        if self.game.had_parsing_errors:
            return chess.pgn.SKIP
        headers = self.game.headers
        is_blitz = (headers["Event"] == "Rated Blitz game")
        is_normal_termination = (headers["Termination"] == "Normal")
        is_time_forfeit_termination = (headers["Termination"] == "Time forfeit")
        has_white_elo = headers["WhiteElo"].isnumeric()
        has_black_elo = headers["BlackElo"].isnumeric()
        is_variant = ("Variant" in headers)
        keep_processing = (is_blitz
                           and (is_normal_termination
                                or is_time_forfeit_termination)
                           and has_white_elo
                           and has_black_elo
                           and not is_variant)
        return None if keep_processing else chess.pgn.SKIP

    def visit_move(self, board, move):
        start_square = chess.SQUARE_NAMES[move.from_square]
        end_square = chess.SQUARE_NAMES[move.to_square]
        piece = chess.piece_symbol(board.piece_type_at(move.from_square))
        if move.promotion:
            promotion = f"={chess.piece_symbol(move.promotion)}"
        else:
            promotion = "-"
        converted_move = f"{piece} {start_square} {end_square} {promotion}"
        self.game.moves.append(converted_move)

    def begin_variation(self):
        return chess.pgn.SKIP

    def handle_error(self, error):
        self.game.had_parsing_errors = True

    def result(self):
        return self.game


class CompressedPgnGameIterator:
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
        game = chess.pgn.read_game(self.text_stream, Visitor=FastGameVisitor)
        if game:
            return game
        else:
            raise StopIteration

    def total_num_bytes_read(self):
        return self.response_raw.num_bytes_read

    def total_num_bytes(self):
        return int(self.response.headers.get("Content-Length", 0))
