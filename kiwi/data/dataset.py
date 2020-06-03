from collections import OrderedDict
from copy import copy
from math import ceil
from pathlib import Path

from more_itertools import nth
from torchtext import data

from kiwi.data.utils import load_vocabularies_to_fields, needs_vocabulary


class KiwiDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        # don't work for pack_padded_sequences
        # return data.interleave_keys(len(ex.source), len(ex.target))
        return len(ex.source)

    # def __init__(self, examples, fields, filter_pred=None):
    #     super().__init__(examples, fields, filter_pred)
    def __init__(
        self, fields, paths, readers=None, lazy=True, filter_pred=None
    ):
        """Create a Corpus by specifying examples and fields.

        Arguments:
            fields: A list of pairs (field name, field object), or a dict.
            paths: A list of paths where to read data from for each field (in
                  the same order as in `fields`), or a dict if `fields` is a
                  dict (with field name as keys).
            readers: A list of functions to be used for reading the files, or a
                     dict with field names as keys. If the function is None or
                     key is missing, the default line by line reading is used.
            lazy: Whether to load data lazily from `files` (true by default).
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.

        Note:
            Both arguments must be lists of the same size (number of fields) or
            dictionaries with the same keys.
        """
        if filter_pred is None:

            def filter_pred(_):
                return True

        self.filter_pred = filter_pred

        self._lazy = lazy
        self.number_of_examples = 0

        self._in_first_loop = True

        self.fields = OrderedDict(fields)
        self.paths = paths
        self.readers = readers
        if isinstance(fields, dict):
            if not isinstance(paths, dict):
                raise TypeError('fields and paths must both be lists or dicts')
            self.paths = [paths[name] for name in self.fields]
            if not isinstance(readers, dict):
                raise TypeError('paths and readers must both be lists or dicts')
            self.readers = [readers.get(name) for name in self.fields]

        self.distinct_fields = OrderedDict()
        for idx, (name, field) in enumerate(self.fields.items()):
            if id(field) not in self.distinct_fields:
                self.distinct_fields[id(field)] = {
                    'names': [name],
                    'idx': [idx],
                    'field': field,
                }
            else:
                self.distinct_fields[id(field)]['names'].append(name)
                self.distinct_fields[id(field)]['idx'].append(idx)

        self.data = []
        for path, reader in zip(self.paths, self.readers):
            if reader:
                if self._lazy:
                    raise NotImplementedError(
                        'Lazy loading is not yet supported with file readers'
                    )
                else:
                    field_data = reader(path)
            else:
                if self._lazy:
                    field_data = Path(path).open('r', encoding='utf8')
                else:
                    with Path(path).open('r', encoding='utf8') as f:
                        field_data = [line.strip() for line in f]
            self.data.append(field_data)

        if not self._lazy:
            nb_lines = [len(fe) for fe in self.data]
            assert min(nb_lines) == max(nb_lines)  # Assert files have same size
            # self._in_first_loop = False
            # self.number_of_examples = len(self.data[0])
            self.examples = [ex for ex in self]

    def inspect(self):
        if self._in_first_loop:
            for _ in self:
                pass

    def fit_vocabularies(
        self, fields_vocab_options, load_vocab=None, extra_datasets=None
    ):
        if extra_datasets is None:
            extra_datasets = []

        if load_vocab:
            load_vocabularies_to_fields(Path(load_vocab), self.fields)

        for field_names_indices in self.distinct_fields.values():
            field = field_names_indices['field']
            names = field_names_indices['names']
            if needs_vocabulary(field):
                kwargs_vocab = fields_vocab_options[names[0]]
                # TODO: sanity check if options are consistent when multiple
                #  fields share vocabulary.
                if 'vectors_fn' in kwargs_vocab:
                    vectors_fn = kwargs_vocab['vectors_fn']
                    kwargs_vocab['vectors'] = vectors_fn()
                    del kwargs_vocab['vectors_fn']

                field.build_vocab(
                    self.field_data(field),
                    *[
                        dataset.field_data(field)
                        for dataset in extra_datasets
                        if id(field) in dataset.distinct_fields
                    ],
                    **kwargs_vocab,
                )

    def field_data(self, field):
        field_names_indices = self.distinct_fields[id(field)]
        field = field_names_indices['field']
        names = field_names_indices['names']
        indices = field_names_indices['idx']

        for (name, idx) in zip(names, indices):
            if self._lazy:
                self.data[idx].seek(0)
            for data_line in self.data[idx]:
                ex = data.Example.fromlist([data_line.strip()], [(name, field)])
                # if self.filter_pred(ex):
                # XXX: above doesn't work because filter_pred accesses more
                #   than one field.
                yield getattr(ex, name)

                if self._in_first_loop:
                    self.number_of_examples += 1
            self._in_first_loop = False

    def __iter__(self):
        if self._lazy:
            for f in self.data:
                f.seek(0)

        for data_line in zip(*self.data):
            data_line = [line.strip() for line in data_line]
            # if not all(data_line):
            #     print('Warning: one of your datasets has a empty line.')
            #     continue
            ex = data.Example.fromlist(data_line, list(self.fields.items()))
            if self.filter_pred(ex):

                yield ex

                if self._in_first_loop:
                    self.number_of_examples += 1
        self._in_first_loop = False

    def __len__(self):
        if self._in_first_loop:
            raise RuntimeError(
                "Haven't yet iterated through dataset once. Be sure to call "
                "`inspect()`"
            )
        return self.number_of_examples

    def __getitem__(self, i):
        if self._lazy:
            data_line = []
            for file in self.data:
                position = file.tell()
                file.seek(0)
                data_line.append(nth(file, i, default=None))
                file.seek(position)
            if not all(data_line):
                raise IndexError('example index out of bounds')
            ex = data.Example.fromlist(data_line, list(self.fields.items()))
            return ex

        return super().__getitem__(i)

    def __getattr__(self, attr):
        if attr in self.fields:
            return self.field_data(attr)

    def __getstate__(self):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        return self.__dict__

    def __setstate__(self, _d):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        return super().__reduce_ex__(proto)

    def split(
        self,
        split_ratio=0.7,
        stratified=False,
        strata_field='label',
        random_state=None,
    ):
        if self._lazy:
            raise NotImplementedError(
                'Split is currently disabled for lazy datasets.'
                # 'Splitting dataset is not supported when loading data lazily.'
            )

        super_dataset = Dataset(
            list(zip(*self.data)), self.fields, filter_pred=None
        )
        datasets = super_dataset.split(
            split_ratio, stratified, strata_field, random_state
        )
        casted_datasets = []
        for dataset in datasets:
            casted_dataset = copy(self)
            casted_dataset.data = dataset.examples
            casted_dataset.number_of_examples = len(dataset.examples)
            casted_datasets.append(casted_dataset)
        return casted_datasets

    def cross_split(self, k):
        """Perform k-fold splits of this dataset.

        That is, split dataset into :math:`k` equally sized chunks and yield
        :math:`k` pairs of datasets with sizes :math:`((k-1), 1)`.

        This is useful for k-fold cross-validation (jackknifing).

        Args:
            k (int): number of folds to split dataset into.

        Yields:
            k pairs of k-1 and 1-fold datasets.

        """
        if self._lazy:
            raise NotImplementedError(
                'Cross-split is currently disabled for lazy datasets.'
                # 'Splitting dataset is not supported when loading data lazily.'
            )

        examples_per_fold = ceil(len(self) / k)
        for fold in range(k):
            held_out_start = examples_per_fold * fold
            held_out_stop = examples_per_fold * (fold + 1)

            held_out_examples = self[held_out_start:held_out_stop]
            held_in_examples = self[:held_out_start] + self[held_out_stop:]

            train_fold = Dataset(held_in_examples, self.fields)
            valid_fold = Dataset(held_out_examples, self.fields)

            yield train_fold, valid_fold
