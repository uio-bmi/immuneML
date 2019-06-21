import collections

from source.environment.LabelConfiguration import LabelConfiguration


class EncoderParams(collections.abc.MutableMapping):

    def __init__(self, result_path: str, label_configuration: LabelConfiguration, model: dict,
                 batch_size: int = 2, learn_model: bool = True, filename: str = ""):

        self.store = {
            "model": model,
            "batch_size": batch_size,
            "learn_model": learn_model,
            "result_path": result_path,
            "label_configuration": label_configuration,
            "filename": filename
        }

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        if key in self.store.keys():
            self.store[key] = value
        else:
            raise RuntimeWarning("Cannot set an arbitrary key for EncoderParams object. Possible keys are: {}"
                                 .format(list(self.store.keys())))

    def __delitem__(self, key):
        raise RuntimeError("Cannot delete the key from EncoderParams object.")

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
