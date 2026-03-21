from .MLPactor import MLPactor
from .RNNActor import RNNActor
from .TransformerActor import TransformerActor

REGISTRY = {}

REGISTRY["mlp"] = MLPactor
REGISTRY["rnn"] = RNNActor
REGISTRY["transformer"] = TransformerActor