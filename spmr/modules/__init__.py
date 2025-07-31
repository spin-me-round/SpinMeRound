from .encoders.modules import GeneralConditioner, GeneralConditionerCamera

UNCONDITIONAL_CONFIG = {
    "target": "spmr.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
