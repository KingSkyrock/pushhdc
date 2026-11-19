"""Loads NysHD's string datasets of different lengths, and splits them.

`protein` is the RhodoPred dataset: rhodopsin proteins from 6 families, as
strings of amino acids. `sms` is the UCI SMS Spam Collection: text messages,
spam or not. Each loads as NysHD's code loads it, so its kernel and our
encoders see the same strings.
"""

import io
import math
import re
import urllib.request
import zipfile
from pathlib import Path
from string import ascii_uppercase

import numpy as np

DATA = Path(__file__).parents[2] / "data"
FAMILIES = (
    "Actinorhodopsin",
    "Bacteriorhodopsin",
    "Halorhodopsin",
    "Proteorhodopsin",
    "Sensory-rhodopsin",
    "Xanthorhodopsin",
)
# the RhodoPred server is gone, so the files come from the Internet Archive's
# copy of it
PROTEIN_URL = (
    "http://web.archive.org/web/20230130193516id_/"
    "https://bioinfo.imtech.res.in/servers/rhodopred/dataset/{}.txt"
)
SMS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
# the letters of both datasets. index `len(LETTERS)` stands for no letter,
# which pads a short string to the length of the longest
LETTERS = ascii_uppercase + "0123456789"
# the `t` that NysHD's code uses for each dataset, and the k-mer length its
# kernel can reach. their code makes a dense row with letters to the power
# 2k numbers for each string, which is too many for k = 3 with these alphabets
ALPHABETS = {"protein": 2, "sms": 3}
LONGEST_K = 2


def load(name: str) -> tuple[list[str], np.ndarray]:
    """the strings and labels of `protein` or `sms`, in the order of NysHD's
    code. the labels are the family numbers, or 1 for spam."""
    if name == "protein":
        return _protein()
    if name == "sms":
        return _sms()
    raise ValueError(f"no dataset {name!r}")


def split(count: int) -> tuple[np.ndarray, np.ndarray]:
    """train and test indices, 80/20. this is the split in NysHD's code (Zhao
    et al., AAAI 2025), so the test strings are the same as theirs."""
    # matches scikit-learn's `train_test_split(test_size=0.2, random_state=42)`
    # so `evolve.py` does not need scikit-learn
    shuffled = np.random.RandomState(42).permutation(count)
    test_count = math.ceil(0.2 * count)
    return np.sort(shuffled[test_count:]), np.sort(shuffled[:test_count])


def padded(strings: list[str]) -> np.ndarray:
    """each string as letter numbers, padded with no letter to the length of
    the longest. letters that are not in `LETTERS` count as no letter."""
    out = np.full(
        (len(strings), max(map(len, strings))), len(LETTERS), dtype=np.int64
    )
    for i, string in enumerate(strings):
        for j, letter in enumerate(string.upper()):
            if letter in LETTERS:
                out[i, j] = LETTERS.index(letter)
    return out


def _protein() -> tuple[list[str], np.ndarray]:
    folder = DATA / "protein"
    strings, labels = [], []
    for family, name in enumerate(FAMILIES):
        path = folder / f"{name}.txt"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(PROTEIN_URL.format(name)) as response:
                path.write_bytes(response.read())
        # each line is a name and a sequence. other lines are headings
        for line in path.read_text().splitlines():
            parts = line.strip().split("::")
            if len(parts) == 2:
                strings.append(parts[1])
                labels.append(family)
    return strings, np.array(labels)


def _sms() -> tuple[list[str], np.ndarray]:
    path = DATA / "sms" / "SMSSpamCollection"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SMS_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read("SMSSpamCollection"))
    strings, labels = [], []
    # each line is a label and a message. NysHD's code keeps only letters and
    # digits, in capitals
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            strings.append(re.sub("[^A-Za-z0-9]+", "", parts[1]).upper())
            labels.append(int(parts[0] == "spam"))
    return strings, np.array(labels)
