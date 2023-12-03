import pyarrow.parquet as pa
import os
import io
import sys


class FileWrapper(io.IOBase):
    def __init__(self, file):
        self.file = file
        self.num_bytes_read = 0
        self.file_size = os.fstat(file.fileno()).st_size

    def read(self, size=-1):
        data = self.file.read(size)
        self.num_bytes_read += sys.getsizeof(data)
        return data

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def flush(self):
        return self.file.flush()

    def close(self):
        return self.file.close()

    def readable(self):
        return self.file.readable()

    def writable(self):
        return False

    def seekable(self):
        return self.file.seekable()

    def isatty(self):
        return self.file.isatty()

    def size(self):
        return self.file_size


class ParquetIterator:
    def __init__(self, file_path, columns=None, batch_size=65536):
        file = open(file_path, "rb")
        self.file_wrapper = FileWrapper(file)
        self.parquet_file = pa.ParquetFile(self.file_wrapper)
        self.batch_size = batch_size
        self.columns = columns

    def __iter__(self):
        return self.parquet_file.iter_batches(batch_size=self.batch_size, 
                                              columns=self.columns)

    def total_num_bytes_read(self):
        return self.file_wrapper.num_bytes_read

    def total_num_bytes(self):
        return self.file_wrapper.size()
