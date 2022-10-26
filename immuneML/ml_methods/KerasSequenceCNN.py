import copy
import logging
from pathlib import Path
import keras
from keras.optimizers import Adam
import numpy as np
import yaml

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.onehot.OneHotSequenceEncoder import OneHotSequenceEncoder
from immuneML.environment.Label import Label
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class KerasSequenceCNN(MLMethod):
    """

    """

    def __init__(self, units_per_layer: list = None, activation: str = None, regularizer: str = None, training_percentage: float = None, result_path: Path = None):

        super().__init__()

        self.units_per_layer = units_per_layer # todo refactor this to something more sensible
        self.activation = activation
        self.regularizer = regularizer
        self.training_percentage = training_percentage

        self.background_probabilities = None
        self.CNN = None
        self.label = None
        self.class_mapping = None
        self.result_path = result_path
        self.feature_names = None

    def predict(self, encoded_data: EncodedData, label: Label):
        predictions_proba = self.predict_proba(encoded_data, label)
        return {label.name: [self.class_mapping[val] for val in (predictions_proba[label.name][:, 1] > 0.5).tolist()]}

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        predictions = self.model.predict(x=encoded_data.examples)
        return {self.label.name: np.hstack([1 - np.array(predictions), predictions])}

    def _create_cnn(self, units_per_layer, input_shape,
               activation, regularizer):
        """ # todo docs reference Mason code
        Generate the CNN layers with a Keras wrapper.

        Parameters
        ---
        units_per_layer: architecture features in list format, i.e.:
            Filter information: [CONV, # filters, kernel size, stride]
            Max Pool information: [POOL, pool size, stride]
            Dropout information: [DROP, dropout rate]
            Flatten: [FLAT]
            Dense layer: [DENSE, number nodes]

        input_shape: a tuple defining the input shape of the data

        activation: Activation function, i.e. ReLU, softmax

        regularizer: Kernel and bias regularizer in convulational and dense
            layers, i.e., regularizers.l1(0.01)
        """

        # Initialize the CNN
        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.InputLayer(input_shape))

        # Build network
        for i, units in enumerate(units_per_layer):
            if units[0] == 'CONV':
                model.add(keras.layers.Conv1D(filters=units[1],
                                              kernel_size=units[2],
                                              strides=units[3],
                                              activation=activation,
                                              kernel_regularizer=regularizer,
                                              bias_regularizer=regularizer,
                                              padding='same'))
            elif units[0] == 'POOL':
                model.add(keras.layers.MaxPool1D(pool_size=units[1],
                                                 strides=units[2]))
            elif units[0] == 'DENSE':
                model.add(keras.layers.Dense(units=units[1],
                                             activation=activation,
                                             kernel_regularizer=regularizer,
                                             bias_regularizer=regularizer))
            elif units[0] == 'DROP':
                model.add(keras.layers.Dropout(rate=units[1]))
            elif units[0] == 'FLAT':
                model.add(keras.layers.Flatten())
            else:
                raise NotImplementedError('Layer type not implemented')

        # Output layer
        # Activation function: Sigmoid
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        return model

    def fit(self, encoded_data: EncodedData, label: Label, optimization_metric=None, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        self.label = label
        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[self.label.name])

        encoded_train_data, encoded_val_data = self._prepare_and_split_data(encoded_data)

        self.model = self._create_cnn(units_per_layer=self.units_per_layer, # todo better input format...
                                      input_shape=encoded_data.examples.shape[1:],
                                      activation=self.activation,
                                      regularizer=self.regularizer)

        self._fit(encoded_train_data=encoded_train_data, encoded_val_data=encoded_val_data)


    def _prepare_and_split_data(self, encoded_data: EncodedData):
        train_indices, val_indices = Util.get_train_val_indices(len(encoded_data.example_ids), self.training_percentage)

        train_data = Util.subset_encoded_data(encoded_data, train_indices)
        val_data = Util.subset_encoded_data(encoded_data, val_indices)

        return train_data, val_data

    def _fit(self, encoded_train_data, encoded_val_data):
        """reference to original code, maybe the input should just be the encoded data instead? #todo"""
        X_train = encoded_train_data.examples
        X_val = encoded_val_data.examples
        y_train = Util.map_to_new_class_values(encoded_train_data.labels[self.label.name], self.class_mapping)
        y_val = Util.map_to_new_class_values(encoded_val_data.labels[self.label.name], self.class_mapping)
        w_train = encoded_train_data.example_weights
        w_val = encoded_val_data.example_weights

        # Compiling the CNN
        opt = Adam(learning_rate=0.000075)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the CNN to the training set # todo return value?? to the log?
        _ = self.model.fit(
            x=X_train, y=y_train, sample_weight=w_train, shuffle=True,
            validation_data=(X_val, y_val, w_val) if w_val is not None else (X_val, y_val),
            epochs=20, batch_size=16, verbose=0
        )

        # print(_.history.keys())
        # print(_.history['val_loss'])

    def fit_by_cross_validation(self, encoded_data: EncodedData, label: Label = None, optimization_metric: str = None,
                                number_of_splits: int = 5, cores_for_training: int = -1):
        logging.warning(f"{KerasSequenceCNN.__name__}: cross_validation is not implemented for this method. Using standard fitting instead...")
        self.fit(encoded_data=encoded_data, label=label)

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)

        self.model.save(path / "model")

        custom_vars = copy.deepcopy(vars(self))
        del custom_vars["model"]
        del custom_vars["result_path"]

        if self.label:
            custom_vars["label"] = vars(self.label)

        params_path = path / "custom_params.yaml"
        with params_path.open('w') as file:
            yaml.dump(custom_vars, file)

    def load(self, path):
        params_path = path / "custom_params.yaml"

        with params_path.open("r") as file:
            custom_params = yaml.load(file, Loader=yaml.SafeLoader)

        for param, value in custom_params.items():
            if hasattr(self, param):
                if param == "label":
                    setattr(self, "label", Label(**value))
                else:
                    setattr(self, param, value)

        self.model = keras.models.load_model(path / "model")

    def check_if_exists(self, path):
        return self.model is not None

    def get_params(self):
        params = copy.deepcopy(vars(self))
        params["model"] = copy.deepcopy(self.model).state_dict()
        return params

    def get_label_name(self):
        return self.label.name

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True

    def get_class_mapping(self) -> dict:
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        return [OneHotEncoder]

    def check_encoder_compatibility(self, encoder):
        """Checks whether the given encoder is compatible with this ML method, and throws an error if it is not."""
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder

        if not issubclass(encoder.__class__, OneHotEncoder):
            raise ValueError(
                f"{encoder.__class__.__name__} is not compatible with ML Method {self.__class__.__name__}. "
                f"Please use one of the following encoders instead: {', '.join([enc_class.__name__ for enc_class in self.get_compatible_encoders()])}")
        else:
            if not isinstance(encoder, OneHotSequenceEncoder):
                raise ValueError(
                    f"{self.__class__.__name__} is only compatible with SequenceDatasets.")

        assert encoder.flatten == False, f"{self.__class__.__name__} is only compatible with OneHotEncoder when setting OneHotEncoder.flatten to False"
        assert encoder.use_positional_info == False, f"{self.__class__.__name__} is only compatible with OneHotEncoder when setting OneHotEncoder.use_positional_info to False"
