from collections import Counter

from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.ReadsType import ReadsType


class KmerFreqReceptorEncoder(KmerFrequencyEncoder):
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset: ReceptorDataset, params: EncoderParams):
        encoded_receptors = []
        receptor_ids = dataset.data.receptor_id
        label_config = params.label_config
        labels = {label: [] for label in label_config.get_labels_by_name()} if params.encode_labels else None

        params.region_type = self.region_type
        sequence_encoder = self._prepare_sequence_encoder()
        feature_names = sequence_encoder.get_feature_names(params)

        # Get chain info from the first receptor to determine chain names
        chains = [Chain.get_chain(chain).name.lower() for chain in dataset.data.chain_pair[0].value]
        
        # Process each chain's sequences
        for chain in chains:
            # Get sequences and counts for this chain
            chain_sequences = getattr(dataset.data, chain)
            chain_counts = chain_sequences.duplicate_count if self.reads == ReadsType.ALL else None
            
            # Try using optimized k-mer computation
            counts = self._compute_kmers(chain_sequences, chain_counts)
            
            if counts is not None:
                # Add chain prefix to all k-mers
                if isinstance(counts, list):
                    chain_counters = [Counter({f"{chain}_{kmer}": count for kmer, count in counter.items()}) 
                                    for counter in counts]
                else:
                    chain_counters = Counter({f"{chain}_{kmer}": count for kmer, count in counts.items()})
            else:
                # Use traditional encoding - should not happen with optimized strategies
                chain_counters = []
                for sequence in chain_sequences:
                    counts = self._encode_sequence(sequence, params, sequence_encoder, Counter())
                    chain_counters.append(Counter({f"{chain}_{kmer}": count for kmer, count in counts.items()}))

            # Store the chain counters for later combining
            if chain == chains[0]:
                encoded_receptors = chain_counters if isinstance(chain_counters, list) else [chain_counters]
            else:
                # Add second chain's k-mers to existing counters
                if isinstance(chain_counters, list):
                    for i, counter in enumerate(chain_counters):
                        encoded_receptors[i].update(counter)
                else:
                    encoded_receptors[0].update(chain_counters)

        if params.encode_labels:
            for label_name in label_config.get_labels_by_name():
                labels[label_name] = getattr(dataset.data, label_name)

        return encoded_receptors, receptor_ids, labels, feature_names

    def _add_chain_to_name(self, count: Counter, chain: str) -> Counter:
        new_counter = Counter()
        for key in count.keys():
            new_counter[f"{chain}_{key}"] = count[key]
        return new_counter
