from pathlib import Path

import numpy as np

import tempfile
import urllib.request
import shutil
import sys
from zipfile import ZipFile
import tarfile

def download_model (url: str, model_path: Path):
    donwload_file(url,
                  filename = model_path.name,
                  out_dir = model_path.parent)

def donwload_file(url:str, filename:str, out_dir:Path):
    assert out_dir.is_dir()
    with tempfile.TemporaryDirectory() as d:
        tmp_file_path = Path(d) / filename
        urllib.request.urlretrieve(url, tmp_file_path)
        shutil.move(str(tmp_file_path), out_dir)

def extract_all(filepath:Path, out_dir:Path):
    assert out_dir.is_dir()
    if filepath.suffix == ".zip":
        with ZipFile(str(filepath), 'r') as zObject:
            zObject.extractall(
                path=out_dir)
    elif filepath.suffix == ".tar" or filepath.suffix == ".gz":
        with tarfile.open(filepath) as f:
            f.extractall(str(out_dir))
    else:
        raise NotImplementedError(f"Unknown extention {filepath.suffix}")

def extract_file(filepath:Path, path_in_tar:str, output_filepath:Path):
    assert filepath.suffix == ".tar" or filepath.suffix == ".gz"
    with tarfile.open(filepath) as f:
        f.extract(path_in_tar, str(output_filepath))

def pad_width (size: int, multiple: int):
    return 0 if size % multiple == 0 else multiple - (size%multiple)

class ImagePadder:
    def __init__(self, multiple, mode):
        self.multiple = multiple
        self.mode = mode

    def pad (self, im: np.ndarray):
        # H,W,C
        rows = im.shape[0]
        cols = im.shape[1]
        self.rows_to_pad = pad_width(rows, self.multiple)
        self.cols_to_pad = pad_width(cols, self.multiple)
        if self.rows_to_pad == 0 and self.cols_to_pad == 0:
            return im
        return np.pad (im, ((0, self.rows_to_pad), (0, self.cols_to_pad), (0, 0)), mode=self.mode)

    def unpad (self, im: np.ndarray):
        w = im.shape[1] - self.cols_to_pad
        h = im.shape[0] - self.rows_to_pad
        return im[:h, :w, :]
