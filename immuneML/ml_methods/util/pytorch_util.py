from torch import Tensor

from immuneML.data_model.bnp_util import write_yaml


def store_weights(model, path):
    state_dict = {key: val.tolist() if isinstance(val, Tensor) else val for key, val in model.state_dict().items()}
    write_yaml(yaml_dict=state_dict, filename=path)