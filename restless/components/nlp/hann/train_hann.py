import sys
# make dep imports work when running as lib / in high-levels scripts
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../..")
try:
    from hann import HierarchicalAttentionNetwork
    from hann import DEFAULT_DATA_PATH, DEFAULT_TRAINING_DATA_PATH, DEFAULT_MODEL_PATH
except:
    from .hann import HierarchicalAttentionNetwork
    from .hann import DEFAULT_DATA_PATH, DEFAULT_TRAINING_DATA_PATH, DEFAULT_MODEL_PATH

# Maps features names to the column numbers from the dataset (CSV format)
# Also sets how much tokenizing we should do (if tokenizing is anything but none,
# then the feature value will be a vector not scalar
pe_headers_feature_keys = [
            # {"name": "e_magic", "index": 0, "tokenize": "char"},
            # {"name": "e_cblp", "index": 1, "tokenize": "char"},
            # {"name": "e_cp", "index": 2, "tokenize": "none"},
            # {"name": "e_crlc", "index": 3, "tokenize": "none"},
            # {"name": "e_cparhdr", "index": 4, "tokenize": "none"},
            {"name": "e_minalloc", "index": 5, "tokenize": "none"},
            {"name": "e_maxalloc", "index": 6, "tokenize": "none"},
            # {"name": "e_ss", "index": 7, "tokenize": "none"},
            # {"name": "e_sp", "index": 8, "tokenize": "none"},
            # {"name": "e_csum", "index": 9, "tokenize": "none"},
            {"name": "e_ip", "index": 10, "tokenize": "none"},
            # {"name": "e_cs", "index": 11, "tokenize": "none"},
            # {"name": "e_lfarlc", "index": 12, "tokenize": "none"},
            # {"name": "e_ovno", "index": 13, "tokenize": "none"},
            # {"name": "e_res", "index": 14, "tokenize": "none"},
            # {"name": "e_oemid", "index": 15, "tokenize": "char"},
            # {"name": "e_oeminfo", "index": 16, "tokenize": "char"},
            # {"name": "e_res2", "index": 17, "tokenize": "none"},
            # {"name": "e_lfanew", "index": 18, "tokenize": "none"},
            {"name": "Machine", "index": 19, "tokenize": "none"},
            {"name": "NumberOfSections", "index": 20, "tokenize": "none"},
            {"name": "PointerToSymbolTable", "index": 22, "tokenize": "none"},
            {"name": "NumberOfSymbols", "index": 23, "tokenize": "none"},
            {"name": "AddressOfEntryPoint", "index": 32, "tokenize": "none"},
            {"name": "CheckSum", "index": 46, "tokenize": "none"}
]

def train_hann(feature_keys: dict, training_fp: str, model_fp: str=DEFAULT_MODEL_PATH):
    # For now the PE header / metadata model will be our default one
    # but eventually we'll have multiple classifiers built using the HANN model
    hann = HierarchicalAttentionNetwork()
    hann.feature_keys = feature_keys
    print("Feature keys: ", hann.feature_keys)
    print("Training fp: ", training_fp)
    model = hann.read_and_train_data(training_fp)
    model.save(model_fp)
    return

if __name__ == "__main__":
    train_hann(pe_headers_feature_keys, DEFAULT_TRAINING_DATA_PATH)
