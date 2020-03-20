from .hann import HANN
from .text_normalizer import TextNormalizer

hann = HANN()
text_normalizer = TextNormalizer()

class NLP:
    """
    Module with all NLP and text processing related features.
    """
    def __init__(self):
        self.hann = hann
        self.text_normalizer = text_normalizer
