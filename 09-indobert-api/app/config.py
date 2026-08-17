import os

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
DEVICE = os.environ.get("DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}