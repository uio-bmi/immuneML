import tensorflow as tf

from immuneML.ml_methods.MLMethod import MLMethod

class RNN_LSTM(MLMethod):
    """
    This is a wrapper of scikit-learn’s LinearSVC class. Please see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
    of LinearSVC for the parameters.

    Note: if you are interested in plotting the coefficients of the SVM model,
    consider running the :ref:`Coefficients` report.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_svm: # user-defined method name
            SVM: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                penalty: l1 # always use penalty l1
                C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under SVM is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_svm: SVM

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"max_iter": 10000, "multi_class": "crammer_singer"}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(RNN_LSTM, self).__init__()

    def _get_ml_model(self, ):

        return tf.keras.layers.LSTM(**params)

    def can_predict_proba(self) -> bool:
        return False

    def get_params(self):
        params = self._parameters
        return params

    def do_random_tests(self):
        inputs = tf.random.normal([32, 10, 8])
        lstm = tf.keras.layers.LSTM(4)
        output = lstm(inputs)
        print(output.shape)
        print(output)
        lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
        print(whole_seq_output.shape)
        print(whole_seq_output)

        print(final_memory_state.shape)
        print(final_memory_state)

        print(final_carry_state.shape)
        print(final_carry_state)

