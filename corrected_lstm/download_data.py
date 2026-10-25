"""Download the original data and require the audited content hash."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

path = Path(__file__).resolve().parent / 'strandenergylist.json'
url = 'https://raw.githubusercontent.com/dereckpiche/dna_aptamers_modelling/main/src/data/strandenergylist.json'
if not path.exists():
    urlretrieve(url, path)
expected = 'fac07d8c27b7b39dc8233e92ddb161710b082e433894cc8cb2ddafafbb30435c'
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print('Dataset checksum verified.')
