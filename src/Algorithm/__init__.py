from .irf_learner import IRFagent
from .coma_learner import COMAagent
from .cds_learner import CDSagent

REGISTRY = {}

REGISTRY["irf"] = IRFagent
REGISTRY["coma"] = COMAagent
REGISTRY["cds"] = CDSagent