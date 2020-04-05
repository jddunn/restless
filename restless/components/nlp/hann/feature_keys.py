import pandas as pd

from .hann import DEFAULT_TRAINING_DATA_PATH

df = pd.read_csv(DEFAULT_TRAINING_DATA_PATH)

# Maps features names to the column numbers from the dataset (CSV format)
# Also sets how much tokenizing we should do (if tokenizing is anything but none,
# then the feature value will be a vector not scalar
pe_headers_feature_keys = [
    {"name": "e_magic", "index": 0, "tokenize": "char"},
    {"name": "e_cblp", "index": 1, "tokenize": "char"},
    {"name": "e_cp", "index": 2, "tokenize": "none"},
    {"name": "e_crlc", "index": 3, "tokenize": "none"},
    {"name": "e_cparhdr", "index": 4, "tokenize": "none"},
    {"name": "e_minalloc", "index": 5, "tokenize": "none"},
    {"name": "e_maxalloc", "index": 6, "tokenize": "none"},
    {"name": "e_ss", "index": 7, "tokenize": "none"},
    {"name": "e_sp", "index": 8, "tokenize": "none"},
    {"name": "e_csum", "index": 9, "tokenize": "none"},
    {"name": "e_ip", "index": 10, "tokenize": "none"},
    {"name": "e_cs", "index": 11, "tokenize": "none"},
    {"name": "e_lfarlc", "index": 12, "tokenize": "none"},
    {"name": "e_ovno", "index": 13, "tokenize": "none"},
    {"name": "e_res", "index": 14, "tokenize": "none"},
    {"name": "e_oemid", "index": 15, "tokenize": "char"},
    {"name": "e_oeminfo", "index": 16, "tokenize": "char"},
    {"name": "e_res2", "index": 17, "tokenize": "none"},
    {"name": "e_lfanew", "index": 18, "tokenize": "none"},
    {"name": "Machine", "index": 19, "tokenize": "none"},
    {"name": "NumberOfSections", "index": 20, "tokenize": "none"},
    {"name": "Machine", "index": 21},
    {"name": "PointerToSymbolTable", "index": 22, "tokenize": "none"},
    {"name": "NumberOfSymbols", "index": 23, "tokenize": "none"},
    {"name": "AddressOfEntryPoint", "index": 32, "tokenize": "none"},
    {"name": "CheckSum", "index": 46, "tokenize": "none"},
]

