from immuneML.util.CompAIRRHelper import CompAIRRHelper


class CompAIRRBatchIterator:
    def __init__(self, paths, sequence_batch_size):
        self.repertoire_ids = None
        self.sequence_batch_size = sequence_batch_size
        self.batch_paths = self.get_batch_dict(paths)
        self.sequence_count = self.compute_sequence_count()

    def __iter__(self):
        return self.get_sequence_vectors(self.repertoire_ids)

    def __len__(self):
        return self.sequence_count

    def compute_sequence_count(self):
        sequence_count = (len(self.batch_paths) - 1) * self.sequence_batch_size

        last_batch_path = self.batch_paths[max(self.batch_paths.keys())]
        sequence_count += len(CompAIRRHelper.read_compairr_output_file(last_batch_path))

        return sequence_count

    def get_batch_dict(self, paths):
        return {self.get_batch_from_path(path): path for path in paths}

    def get_batch_from_path(self, path):
        return int(path.stem.split("_batch")[1])

    def set_repertoire_ids(self, repertoire_ids):
        self.repertoire_ids = repertoire_ids

    def get_batches(self, repertoire_ids = None):
        for batch_idx in sorted(self.batch_paths):
            path = self.batch_paths[batch_idx]
            batch = CompAIRRHelper.read_compairr_output_file(path)

            if repertoire_ids is not None:
                batch = batch[repertoire_ids]

            # count clones only
            batch[batch > 1] = 1
            batch.sort_index(inplace=True)

            yield batch

    def get_sequence_vectors(self, repertoire_ids = None):
        for batch in self.get_batches(repertoire_ids):
            for idx, sequence_vector in batch.iterrows():
                yield sequence_vector.to_numpy()


