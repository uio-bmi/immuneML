



ApplyGenModel instruction implements applying generative AIRR models on the sequence level.

This instruction takes as input a trained model (trained in the :ref:`TrainGenModel` instruction)
which will be used for generating data and the number of sequences to be generated.
It can also produce reports of the applied model and reports of generated sequences.

Optionally, one or more datasets can be provided (p_gen_datasets) to compute the generation probabilities of their
sequences under the trained model. The sequences of each dataset are then stored along with their generation
probabilities (column p_gen_<model name>) in a separate tsv file in the datasets_with_p_gens folder. This is only available for
models that can compute generation probabilities.


**Specification arguments:**

- gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model; it can
  be set to 0 to skip generation if only the generation probabilities of the provided datasets are needed

- reports (list): list of report ids (defined under definitions/reports) to apply after generating
  gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
  on the fitted model)

- ml_config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

- p_gen_datasets (list): optional; names of the sequence datasets (defined under definitions/datasets) for which
  the generation probabilities under the trained model should be computed; if not specified, no generation
  probabilities are computed

**YAML specification:**

.. highlight:: yaml
.. code-block:: yaml

    instructions:
        my_apply_gen_model_inst: # user-defined instruction name
            type: ApplyGenModel
            gen_examples_count: 100
            ml_config_path: ./config.zip
            reports: [data_rep1, ml_rep2]
            p_gen_datasets: [my_dataset1, my_dataset2] # optional


