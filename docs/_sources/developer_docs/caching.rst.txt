

To prevent recomputing the same result a second time, immuneML uses caching.
Caching can be applied to methods which compute an (intermediate) result.
The result is stored to a file, and when the same method call is made, the previously
stored result is retrieved from the file and returned.

We recommend applying caching to methods which are computationally expensive and may be called
multiple times in the same way. For example, encoders are a good target for caching as they
may take long to compute and can be called multiple times on the same data when combined
with different ML methods. ML methods typically do not require caching, as you would
want to fit ML methods with different parameters or to differently encoded data.


Any method call in immuneML can be cached as follows:

.. code:: python

    result = CacheHandler.memo_by_params(params = cache_params, fn = lambda: my_method_for_caching(my_method_param1, my_method_param2, ...))


The :code:`CacheHandler.memo_by_params` method does the following:

- Using the caching parameters, a unique cache key (string) is created.
- CacheHandler checks if there already exists a previously computed result that is associated with this key.
- If the result exists, the result is returned without (re)computing the method.
- If the result does not exist, the method is computed, its result is stored using the cache key, and the result is returned.


The :code:`lambda` function call simply calls the method whose results will be cached, using any required parameters.
The :code:`cache_params` represent the unique, immutable parameters used to compute the cache key.
It should have the following properties:

- It must be a nested tuple containing *only* immutable items such as strings, booleans and integers.
  It cannot contain mutable items like lists, dictionaries, sets and objects (they all need to be converted nested tuples of immutable items).
- It should include *every* factor that can contribute to a difference in the results of the computed method.
  For example, when caching the encode_data step, the following should be included:

  - dataset descriptors (dataset id, example ids, dataset type),
  - encoding name,
  - labels,
  - :code:`EncoderParams.learn_model` if used,
  - all relevant input parameters to the encoder. Preferentially retrieved automatically (such as by :code:`vars(self)`),
    as this ensures that if new parameters are added to the encoder, they are always added to the caching params.

For example, :py:obj:`~immuneML.encodings.onehot.OneHotEncoder.OneHotEncoder` computes its
caching parameters as follows:

  .. code:: python

    def _prepare_caching_params(self, dataset, params: EncoderParams):
        return (("dataset_identifier", dataset.identifier),
                ("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_type", dataset.__class__.__name__),
                ("encoding", OneHotEncoder.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding_params", tuple(vars(self).items())))

The construction of caching parameters must be done carefully, as caching bugs are extremely difficult
to discover. Rather add 'too much' information than too little.
A missing parameter will not lead to an error, but can result in silently copying over
results from previous method calls that do not match the current call.
