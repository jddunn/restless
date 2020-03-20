from .hann import HierarchicalAttentionNetwork
from .text_normalizer import TextNormalizer

hann = HierarchicalAttentionNetwork()
text_normalizer = TextNormalizer()

class NLP:
    """
    Module with all NLP and text processing related features.
    """

    def __init__(self):
        self.hann = hann
        self.text_normalizer = text_normalizer
