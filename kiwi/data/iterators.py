#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import math
import random

import torch
from torchtext import data as tt_data
from torchtext.data import Batch, BucketIterator, Dataset, Iterator


class NamedIterator(Iterator):
    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield NamedBatch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return


class BucketNamedIterator(BucketIterator, NamedIterator):
    def create_batches(self):
        if self.sort:
            self.batches = better_batch(
                self.data(),
                self.batch_size,
                self.batch_size_fn,
            )
        else:
            self.batches = better_pool(
                self.data(),
                self.batch_size,
                self.sort_key,
                self.batch_size_fn,
                random_shuffler=self.random_shuffler,
                shuffle=self.shuffle,
                sort_within_batch=self.sort_within_batch,
            )


class LazyIterator(Iterator):
    """
    Consume a generator for a specific number of steps (`buffer_size`), storing
    the examples in a buffer. The iterator will be built using this buffer.

    Args:
        buffer_size(int): The number of examples to be stored in the buffer.
            Default: batch_size * 1024
    """

    def __init__(self, *args, buffer_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []
        if buffer_size is None:
            # Minibatches will have the same size if buffer_size
            # is divisible by batch_size
            buffer_size = self.batch_size * 2 ** 10
        self.buffer_size = buffer_size
        self.batches = None

    def data(self):
        return iter(self.dataset)

    def clear_buffer(self):
        self.buffer.clear()

    def prepare_buffer(self):
        if self.sort:
            self.buffer.sort(key=self.sort_key)
        elif self.shuffle:
            buffer_size = range(len(self.buffer))
            self.buffer = [
                self.buffer[i] for i in self.random_shuffler(buffer_size)
            ]

    def create_batches(self):
        self.batches = tt_data.batch(
            self.buffer, self.batch_size, self.batch_size_fn
        )

    def consume_buffer(self):
        self.prepare_buffer()
        self.create_batches()
        for minibatch in self.batches:
            self.iterations += 1
            self._iterations_this_epoch += 1
            if self.sort_within_batch:
                if self.sort:
                    minibatch.reverse()
                else:
                    minibatch.sort(key=self.sort_key, reverse=True)
            yield NamedBatch(minibatch, self.dataset, self.device)

    def __iter__(self):
        while True:
            self.init_epoch()
            self.clear_buffer()

            for ex in self.data():
                self.buffer.append(ex)
                if len(self.buffer) == self.buffer_size:
                    for batch in self.consume_buffer():
                        yield batch
                    self.clear_buffer()

            # in case the buffer is not empty
            if len(self.buffer) > 0:
                for batch in self.consume_buffer():
                    yield batch
                self.clear_buffer()

            if not self.repeat:
                return


class LazyBucketIterator(BucketIterator, LazyIterator):
    def create_batches(self):
        if self.sort:
            self.batches = tt_data.batch(
                self.buffer, self.batch_size, self.batch_size_fn
            )
        else:
            self.batches = better_pool(
                self.buffer,
                self.batch_size,
                self.sort_key,
                self.batch_size_fn,
                random_shuffler=self.random_shuffler,
                shuffle=self.shuffle,
                sort_within_batch=self.sort_within_batch,
                lookahead=self.buffer_size,
            )


def better_batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:

        def batch_size_fn(new, count, sofar):
            return count

    buffered_minibatch = []
    next_minibatch = []
    size_so_far = 0
    buffered_size_so_far = 0
    for ex in data:
        if buffered_size_so_far < batch_size:
            buffered_minibatch.append(ex)
            buffered_size_so_far = batch_size_fn(
                ex, len(buffered_minibatch), buffered_size_so_far
            )
        else:
            next_minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(next_minibatch), size_so_far)
            if size_so_far == batch_size:
                yield buffered_minibatch
                buffered_minibatch = next_minibatch
                next_minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield buffered_minibatch
                oversize = size_so_far - batch_size
                buffered_minibatch = next_minibatch[:-oversize]
                next_minibatch = next_minibatch[-oversize:]
                size_so_far = batch_size_fn(ex, oversize, 0)

    if not buffered_minibatch:
        raise Exception('We should not have lost a buffered minibatch!')

    if len(next_minibatch) == 1:  # deal with edge case
        mid_point = (len(buffered_minibatch) + 1) // 2
        next_minibatch = buffered_minibatch[mid_point:] + next_minibatch
        buffered_minibatch = buffered_minibatch[:mid_point]

    yield buffered_minibatch
    if next_minibatch:
        yield next_minibatch


def better_pool(
    data,
    batch_size,
    key,
    batch_size_fn=lambda new, count, sofar: count,
    random_shuffler=None,
    shuffle=False,
    sort_within_batch=False,
    lookahead=100,
):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size lookahead*batch_size, sorts examples
    within each chunk using sort_key, then batch these examples and shuffle the
    batches.

    This is an improved version over torchtext.data.pool.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in better_batch(data, batch_size * lookahead, batch_size_fn):
        if sort_within_batch:
            p = sorted(p, key=key)
        p_batch = better_batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in p_batch:
                yield b


def build_bucket_iterator(
    dataset, device, batch_size, is_train, lazy=False, buffer_size=None
):
    # train_iter = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=is_train,
    #     # num_workers=4,
    #     collate_fn=torchtext_collate,
    # )

    device_obj = None if device is None else torch.device(device)
    iterator_cls = BucketNamedIterator
    kwargs = {}
    if lazy:
        iterator_cls = LazyBucketIterator
        kwargs = dict(buffer_size=buffer_size)

    iterator = iterator_cls(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        sort_key=None, # dataset.sort_key,
        sort=False,
        # sorts the data within each minibatch in decreasing order
        # set to true if you want use pack_padded_sequences
        sort_within_batch=False,
        # shuffle batches
        shuffle=False,
        device=device_obj,
        train=is_train,
        **kwargs
    )
    return iterator


# def torchtext_collate(batch):
#     """Slightly different from default_collate: add torchtext.data.Batch to
#        it.
#        Puts each data field into a tensor with outer dimension batch size.
#        From https://github.com/pytorch/text/issues/283.
#     """
#
#     error_msg = "batch must contain tensors, numbers, dicts or lists; found "
#                 "{}"
#     elem_type = type(batch[0])
#     if isinstance(batch[0], torch.Tensor):
#         out = None
#         if _use_shared_memory:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = batch[0].storage()._new_shared(numel)
#             out = batch[0].new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#         and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if re.search('[SaUO]', elem.dtype.str) is not None:
#                 raise TypeError(error_msg.format(elem.dtype))
#             return torch.stack([torch.from_numpy(b) for b in batch], 0)
#         if elem.shape == ():  # scalars
#             py_type = float if elem.dtype.name.startswith('float') else int
#             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#     elif isinstance(batch[0], int_classes):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], float):
#         return torch.DoubleTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
#     elif isinstance(batch[0], collections.Mapping):
#         return {key: torchtext_collate([d[key] for d in batch]) for key in
#         batch[0]}
#     elif isinstance(batch[0], torchtext.data.Batch):  # difference here
#         return {key: torchtext_collate([getattr(d, key) for d in batch]) for
#         key in batch[0].dataset.fields.keys()}
#     elif isinstance(batch[0], collections.Sequence):
#         transposed = zip(*batch)
#         return [torchtext_collate(samples) for samples in transposed]
#
#     raise TypeError((error_msg.format(type(batch[0]))))


class NamedBatch(Batch):
    def _get_field_named_values(self, fields):
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return {fields[0]: getattr(self, fields[0])}
        else:
            return {f: getattr(self, f) for f in fields}

    def __iter__(self):
        yield self._get_field_named_values(self.input_fields)
        yield self._get_field_named_values(self.target_fields)

    def __contains__(self, item):
        return hasattr(self, item)

    def __getitem__(self, item):
        return getattr(self, item)

    def named_tensors(self):
        inputs, outputs = (b for b in self)
        return inputs, outputs
