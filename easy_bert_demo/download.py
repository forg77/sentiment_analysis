import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base")