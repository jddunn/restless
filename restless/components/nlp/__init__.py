import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from .nlp import NLP

from .text_normalizer import TextNormalizer

text_normalizer = TextNormalizer()
