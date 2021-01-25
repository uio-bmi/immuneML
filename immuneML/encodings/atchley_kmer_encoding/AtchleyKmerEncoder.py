import logging
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, List

import numpy as np
import yaml

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.atchley_kmer_encoding.RelativeAbundanceType import RelativeAbundanceType
from immuneML.encodings.atchley_kmer_encoding.Util import Util
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from scripts.specification_util import update_docs_per_mapping


class AtchleyKmerEncoder(DatasetEncoder):
    """
    Represents a repertoire through Atchley factors and relative abundance of k-mers. Should be used in combination
    with the :ref:`AtchleyKmerMILClassifier`.

    For more details, see the original publication: Ostmeyer J,
    Christley S, Toby IT, Cowell LG. Biophysicochemical motifs in T cell receptor sequences distinguish repertoires from tumor-infiltrating
    lymphocytes and adjacent healthy tissue. Cancer Res. Published online January 1, 2019:canres.2292.2018. `doi:10.1158/0008-5472.CAN-18-2292
    <https://cancerres.aacrjournals.org/content/79/7/1671>`_ .

    Note that sequences in the repertoire with length shorter than skip_first_n_aa + skip_last_n_aa + k will not be encoded.

    Arguments:

        k (int): k-mer length

        skip_first_n_aa (int): number of amino acids to remove from the beginning of the receptor sequence

        skip_last_n_aa (int): number of amino acids to remove from the end of the receptor sequence

        abundance: how to compute abundance term for k-mers

        normalize_all_features (bool): when normalizing features to have 0 mean and unit variance, this parameter indicates if the abundance
        feature should be included in the normalization

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_encoder:
                AtchleyKmer:
                    k: 4
                    skip_first_n_aa: 3
                    skip_last_n_aa: 3
                    abundance: RELATIVE_ABUNDANCE
                    normalize_all_features: False

    """

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            return AtchleyKmerEncoder(**params)
        else:
            raise ValueError(f"AtchleyKmerEncoder can only be applied to repertoire dataset, got {type(dataset).__name__} instead.")

    def __init__(self, k: int, skip_first_n_aa: int, skip_last_n_aa: int, abundance: str, normalize_all_features: bool, name: str = None):
        location = "AtchleyKmerEncoder"
        ParameterValidator.assert_type_and_value(k, int, location, "k", 1)
        ParameterValidator.assert_type_and_value(skip_first_n_aa, int, location, "skip_first_n_aa", 0)
        ParameterValidator.assert_type_and_value(skip_last_n_aa, int, location, "skip_last_n_aa", 0)
        ParameterValidator.assert_in_valid_list(abundance.upper(), [ab.name for ab in RelativeAbundanceType], location, "abundance")
        ParameterValidator.assert_type_and_value(normalize_all_features, bool, location, "normalize_all_features")
        self.k = k
        self.skip_first_n_aa = skip_first_n_aa
        self.skip_last_n_aa = skip_last_n_aa
        self.abundance = RelativeAbundanceType[abundance.upper()]
        self.normalize_all_features = normalize_all_features
        self.name = name
        self.scaler_path = None
        self.vectorizer_path = None

    def encode(self, dataset, params: EncoderParams):

        examples, keys, labels = self._encode_examples(dataset, params)
        examples, kmer_keys = self._vectorize_examples(examples, params, keys)

        # normalize to zero mean and unit variance only features coming from Atchley factors
        tmp_examples = examples[:, :, :-1] if not self.normalize_all_features else examples
        flattened_vectorized_examples = tmp_examples.reshape(examples.shape[0] * examples.shape[1], -1)
        if self.scaler_path is None:
            self.scaler_path = params.result_path / "atchley_factor_scaler.pickle"
        scaled_examples = FeatureScaler.standard_scale(self.scaler_path, flattened_vectorized_examples)
        if hasattr(scaled_examples, "todense"):
            scaled_examples = scaled_examples.todense()

        if self.normalize_all_features:
            examples = np.array(scaled_examples).reshape(examples.shape[0], len(kmer_keys), -1)
        else:
            examples[:, :, :-1] = np.array(scaled_examples).reshape(examples.shape[0], len(kmer_keys), -1)

        # swap axes to get examples x atchley_factors x kmers dimensions
        examples = np.swapaxes(examples, 1, 2)

        feature_names = [f"atchley_factor_{j}_aa_{i}" for i in range(1, self.k+1) for j in range(1, Util.ATCHLEY_FACTOR_COUNT+1)] + ["abundance"]
        encoded_data = EncodedData(examples=examples, example_ids=dataset.get_example_ids(), feature_names=feature_names, labels=labels,
                                   encoding=AtchleyKmerEncoder.__name__, info={"kmer_keys": kmer_keys})

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset: RepertoireDataset, params: EncoderParams) -> Tuple[list, set, dict]:

        keys = set()
        example_count = dataset.get_example_count()

        arguments = [(repertoire, index, example_count) for index, repertoire in enumerate(dataset.repertoires)]

        with Pool(params.pool_size) as pool:
            chunksize = math.floor(dataset.get_example_count() / params.pool_size) + 1
            examples = pool.starmap(self._process_repertoire_cached, arguments, chunksize=chunksize)

        for example in examples:
            keys.update(list(example.keys()))

        labels = dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None

        return examples, keys, labels

    def _process_repertoire_cached(self, repertoire, index, example_count):
        return CacheHandler.memo_by_params((('repertoire', repertoire.identifier), ('encoder', AtchleyKmerEncoder.__name__),
                                            (self.abundance, self.skip_last_n_aa, self.skip_first_n_aa, self.k)),
                                           lambda: self._process_repertoire(repertoire, index, example_count), CacheObjectType.ENCODING_STEP)

    def _process_repertoire(self, repertoire, index, example_count):
        if self.skip_first_n_aa > 0 and self.skip_last_n_aa > 0:
            remove_aa_func = lambda seqs: [seq[self.skip_first_n_aa:-self.skip_last_n_aa] for seq in seqs]
        elif self.skip_last_n_aa > 0:
            remove_aa_func = lambda seqs: [seq[:-self.skip_last_n_aa] for seq in seqs]
        else:
            remove_aa_func = lambda seqs: [seq[self.skip_first_n_aa:] for seq in seqs]

        logging.info(f"AtchleyKmerEncoder: encoding repertoire {index + 1}/{example_count}.")

        sequences, counts = self._trunc_sequences(repertoire, remove_aa_func)
        abundances = Util.compute_abundance(sequences, counts, self.k, self.abundance)
        kmers = list(abundances.keys())
        atchley_factors_df = Util.get_atchely_factors(kmers, self.k)
        atchley_factors_df["abundances"] = np.log(list(abundances.values()))

        encoded = atchley_factors_df.to_dict('index')
        encoded = {key: list(encoded[key].values()) for key in encoded}
        return encoded

    def _vectorize_examples(self, examples, params: EncoderParams, keys: set) -> Tuple[np.ndarray, list]:

        if self.vectorizer_path is None:
            self.vectorizer_path = params.result_path / "vectorizer_keys.yaml"

        if params.learn_model is True:
            kmer_keys = sorted(list(keys))
            PathBuilder.build(params.result_path)
            with self.vectorizer_path.open("w") as file:
                yaml.dump(kmer_keys, file)
        else:
            with self.vectorizer_path.open("r") as file:
                kmer_keys = yaml.safe_load(file)

        vectorized_examples = [
            np.array([np.array(example[key]) if key in example else np.zeros(self.k * Util.ATCHLEY_FACTOR_COUNT + 1) for key in kmer_keys])
            for example in examples]
        return np.array(vectorized_examples, dtype=np.float32), kmer_keys

    def _trunc_sequences(self, repertoire, remove_aa_func):
        sequences = repertoire.get_sequence_aas()
        counts = repertoire.get_counts()
        indices = [i for i in range(sequences.shape[0]) if len(sequences[i]) >= self.skip_first_n_aa + self.skip_last_n_aa + self.k]
        sequences = sequences[indices]
        counts = counts[indices]
        if self.skip_first_n_aa > 0 or self.skip_last_n_aa > 0:
            sequences = np.apply_along_axis(remove_aa_func, 0, sequences)
        return sequences, counts

    def get_additional_files(self) -> List[str]:
        return [self.scaler_path, self.vectorizer_path]

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        for attribute in ["scaler_path", "vectorizer_path"]:
            encoder = DatasetEncoder.load_attribute(encoder, encoder_file, attribute)
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(AtchleyKmerEncoder.__doc__)

        valid_values = str([item.name for item in RelativeAbundanceType])[1:-1].replace("'", "`")
        mapping = {
            "how to compute abundance term for k-mers":
                f"how to compute abundance term for k-mers; valid values are {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
