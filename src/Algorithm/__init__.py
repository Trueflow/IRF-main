from .liir_learner import LIIRagent
from .mapoca_learner import POCAagent
from .coma_learner import COMAagent
from .cds_learner import CDSagent
from .ppo_learner import PPOagent
from .emc_learner import EMCagent

REGISTRY = {}

REGISTRY["liir"] = LIIRagent
REGISTRY["coma"] = COMAagent
REGISTRY["poca"] = POCAagent
REGISTRY["cds"] = CDSagent
REGISTRY["emc"] = EMCagent

# single
REGISTRY["ppo"] = PPOagent