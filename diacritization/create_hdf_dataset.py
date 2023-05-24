import argparse
import os
import h5py
import numpy
from .dictionary import (
    arabic_dictionary,
    harakat,
    harakat_index_dict,
    buckwalter_dictionary,
    harakat_buckwalter,
    arabic_index_dict,
)


def hdf5_strings(handle, name, data):
    """
    :param h5py.File handle:
    :param str name:
    :param numpy.ndarray data:
    """
    # noinspection PyBroadException
    try:
        s = max([len(d) for d in data])
        dset = handle.create_dataset(name, (len(data),), dtype="S" + str(s))
        dset[...] = data
    except Exception:
        # noinspection PyUnresolvedReferences
        dt = h5py.special_dtype(vlen="unicode")
        del handle[name]
        dset = handle.create_dataset(name, (len(data),), dtype=dt)
        dset[...] = data


class SimpleHDFWriter:
    def __init__(self, filename, dim, labels=None, ndim=None):
        """
        :param str filename:
        :param int|None dim:
        :param int ndim: counted without batch
        :param numpy.ndarray|List[str]|None labels:
        """
        if ndim is None:
            if dim is None:
                ndim = 1
            else:
                ndim = 2
        self.dim = dim
        self.ndim = ndim
        self.labels = labels
        if labels:
            assert len(labels) == dim
        self._file = h5py.File(filename, "w")

        self._file.attrs["numTimesteps"] = 0  # we will increment this on-the-fly
        self._other_num_time_steps = 0
        self._file.attrs["inputPattSize"] = dim or 1
        self._file.attrs["numDims"] = 1  # ignored?
        self._file.attrs["numLabels"] = dim or 1
        self._file.attrs["numSeqs"] = 0  # we will increment this on-the-fly
        if labels:
            hdf5_strings(self._file, "labels", labels)
        else:
            self._file.create_dataset("labels", (0,), dtype="S5")

        self._datasets = {}  # type: dict[str, h5py.Dataset]
        self._tags = []  # type: list[str]
        self._seq_lengths = self._file.create_dataset("seqLengths", (0, 2), dtype="i", maxshape=(None, 2))

    def _insert_h5_inputs(self, raw_data):
        """
        Inserts a record into the hdf5-file.
        Resizes if necessary.

        :param numpy.ndarray raw_data: shape=(time,data) or shape=(time,)
        """
        assert raw_data.ndim >= 1
        name = "inputs"
        if name not in self._datasets:
            self._datasets[name] = self._file.create_dataset(
                name,
                raw_data.shape,
                raw_data.dtype,
                maxshape=tuple(None for _ in raw_data.shape),
            )
        else:
            old_shape = self._datasets[name].shape
            self._datasets[name].resize((old_shape[0] + raw_data.shape[0],) + old_shape[1:])
        # append raw data to dataset
        self._datasets[name][self._file.attrs["numTimesteps"] :] = raw_data
        self._file.attrs["numTimesteps"] += raw_data.shape[0]
        self._file.attrs["numSeqs"] += 1

    def _insert_h5_other(self, data_key, raw_data, dtype=None, add_time_dim=False, dim=None):
        """
        :param str data_key:
        :param numpy.ndarray|int|float|list[int] raw_data: shape=(time,data) or shape=(time,) or shape=()...
        :param str dtype:
        :param bool add_time_dim:
        :param int|None dim:
        """
        if isinstance(raw_data, (int, float, list)):
            raw_data = numpy.array(raw_data)
        assert isinstance(raw_data, numpy.ndarray)
        if add_time_dim:
            raw_data = raw_data[None, :]
        assert raw_data.ndim > 0 and raw_data.shape[0] > 0
        if dtype:
            raw_data = raw_data.astype(dtype)
        if dim is None:
            if raw_data.ndim > 1:
                dim = raw_data.shape[-1]
            else:
                dim = 1  # dummy
        assert data_key != "inputs"
        name = data_key
        # Keep consistent with _insert_h5_inputs.
        if name not in self._datasets:
            if "targets/data" not in self._file:
                self._file.create_group("targets/data")
            if "targets/size" not in self._file:
                self._file.create_group("targets/size")
            if "targets/labels" not in self._file:
                self._file.create_group("targets/labels")
            hdf5_strings(self._file, "targets/labels/%s" % data_key, ["dummy-label"])
            self._datasets[name] = self._file["targets/data"].create_dataset(
                data_key,
                raw_data.shape,
                raw_data.dtype,
                maxshape=tuple(None for _ in raw_data.shape),
            )
            self._file["targets/size"].attrs[data_key] = [
                dim,
                raw_data.ndim,
            ]  # (dim, ndim)
        else:
            old_shape = self._datasets[name].shape
            self._datasets[name].resize(
                (old_shape[0] + raw_data.shape[0],)
                + tuple(max(old, new) for old, new in zip(old_shape[1:], raw_data.shape[1:]))
            )

        assert (
            self._file.attrs["numSeqs"] > 0 and self._seq_lengths.shape[0] > 0
        )  # assume _insert_h5_inputs called before

        if self._seq_lengths[self._file.attrs["numSeqs"] - 1, 1]:
            assert self._seq_lengths[self._file.attrs["numSeqs"] - 1, 1] == raw_data.shape[0]
        else:
            self._seq_lengths[self._file.attrs["numSeqs"] - 1, 1] = raw_data.shape[0]
            self._other_num_time_steps += raw_data.shape[0]

        offset = self._other_num_time_steps - raw_data.shape[0]
        hdf_data = self._datasets[name]
        hdf_data[offset:] = raw_data

    def insert_batch(self, inputs, seq_len, seq_tag, extra=None):
        """
        :param numpy.ndarray inputs: shape=(n_batch,time,data) (or (n_batch,time), or (n_batch,time1,time2), ...)
        :param list[int]|dict[int,list[int]|numpy.ndarray] seq_len: sequence lengths (per axis, excluding batch axis)
        :param list[str|bytes] seq_tag: sequence tags of length n_batch
        :param dict[str,numpy.ndarray]|None extra:
        """
        n_batch = len(seq_tag)
        assert n_batch == inputs.shape[0]
        assert inputs.ndim == self.ndim + 1  # one more for the batch-dim
        if not isinstance(seq_len, dict):
            seq_len = {0: seq_len}
        assert isinstance(seq_len, dict)
        assert all(
            [isinstance(key, int) and isinstance(value, (list, numpy.ndarray)) for (key, value) in seq_len.items()]
        )
        ndim_with_seq_len = self.ndim - (1 if self.dim else 0)
        assert all([0 <= key < ndim_with_seq_len for key in seq_len.keys()]) or ndim_with_seq_len == 0
        assert len(seq_len) == ndim_with_seq_len
        assert all([n_batch == len(value) for (key, value) in seq_len.items()])
        assert all([max(value) == inputs.shape[key + 1] for (key, value) in seq_len.items()])
        if self.dim:
            assert self.dim == inputs.shape[-1]
        if extra:
            assert all([n_batch == value.shape[0] for value in extra.values()])

        seqlen_offset = self._seq_lengths.shape[0]
        self._seq_lengths.resize(seqlen_offset + n_batch, axis=0)

        for i in range(n_batch):
            self._tags.append(seq_tag[i])
            # Note: Currently, our HDFDataset does not support to have multiple axes with dynamic length.
            # Thus, we flatten all together, and calculate the flattened seq len.
            # (Ignore this if there is only a single time dimension.)
            flat_seq_len = numpy.prod([seq_len[axis][i] for axis in range(ndim_with_seq_len)])
            assert flat_seq_len > 0
            flat_shape = [flat_seq_len]
            if self.dim:
                flat_shape.append(self.dim)
            self._seq_lengths[seqlen_offset + i, 0] = flat_seq_len
            data = inputs[i]
            data = data[tuple([slice(None, seq_len[axis][i]) for axis in range(ndim_with_seq_len)])]
            data = numpy.reshape(data, flat_shape)
            self._insert_h5_inputs(data)
            if len(seq_len) > 1:
                # Note: Because we have flattened multiple axes with dynamic len into a single one,
                # we want to store the individual axes lengths. We store those in a separate data entry "sizes".
                # Note: We could add a dummy time-dim for this "sizes", and then have a feature-dim = number of axes.
                # However, we keep it consistent to how we handled it in our 2D MDLSTM experiments.
                self._insert_h5_other(
                    "sizes",
                    [seq_len[axis][i] for axis in range(ndim_with_seq_len)],
                    add_time_dim=False,
                    dtype="int32",
                )
            if extra:
                for key, value in extra.items():
                    self._insert_h5_other(key, value[i])

    def close(self):
        max_tag_len = max([len(d) for d in self._tags]) if self._tags else 0
        self._file.create_dataset("seqTags", shape=(len(self._tags),), dtype="S%i" % (max_tag_len + 1))
        for i, tag in enumerate(self._tags):
            self._file["seqTags"][i] = numpy.array(tag, dtype="S%i" % (max_tag_len + 1))
        self._file.close()


def main():
    parser = argparse.ArgumentParser(description="create hdf files to be used in Returnn")

    parser.add_argument("training_text", help="path to the arabic diacritized text (with diacritics included)")
    parser.add_argument("hdf_source_letter_dataset", help="path to the source (letter) hdf dataset to be created")
    parser.add_argument(
        "hdf_source_diacritic_dataset",
        help="path to the source (diacritics) hdf dataset to be created",
    )
    parser.add_argument("hdf_target_dataset", help="path to the target hdf dataset to be created")
    parser.add_argument(
        "--masking_factor",
        default=1.0,
        help="percent to randomly mask diacritics. " "1.0 means to remove 100% of diacritics, 0.5 means 50%, so on.",
    )
    parser.add_argument(
        "--max_char_seq",
        default=1500,
        help="maximum length in terms of the number of characters.",
    )
    args = parser.parse_args()

    masking_factor = float(args.masking_factor)
    max_char_seq = int(args.max_char_seq)

    hdf_source_letter_writer = SimpleHDFWriter(args.hdf_source_letter_dataset, dim=None)
    hdf_source_diacritic_writer = SimpleHDFWriter(args.hdf_source_diacritic_dataset, dim=None)
    hdf_target_writer = SimpleHDFWriter(args.hdf_target_dataset, dim=None)
    filtered_segments = open("filtered_segments", "w")
    sequence_name = os.path.splitext(os.path.basename(args.hdf_source_letter_dataset))[0]

    source_data = open(args.training_text, "rt")

    inverse_buckwalter_dict = {}
    for i, char in enumerate(buckwalter_dictionary):
        inverse_buckwalter_dict[i] = char

    inverse_harakat_dict = {0: "_"}
    for i, char in enumerate(harakat_buckwalter):
        inverse_harakat_dict[i + 1] = char

    buckwalter_transform_dict = {}
    for arab, buck in zip(arabic_dictionary, buckwalter_dictionary):
        buckwalter_transform_dict[arab] = buck

    for i, source_line in enumerate(source_data):
        source_indices = []
        target_indices = []
        print(i)
        for char in source_line.strip():
            if char in harakat:
                if len(target_indices) == 0:
                    # sentence start with a harakat, skip
                    continue
                if target_indices[-1] != 0:
                    # double harakat with shadda in front
                    target_indices[-1] = len(harakat) + harakat_index_dict[char]
                else:
                    target_indices[-1] = harakat_index_dict[char]
            else:
                source_indices.append(arabic_index_dict.get(char, 0))
                target_indices.append(0)

        # randomly select a percent of a list and set it to zero
        source_diac_indices = target_indices.copy()
        before_masking = [p for p, e in enumerate(source_diac_indices) if e != 0]

        # Due to the mask size
        if masking_factor == 1.0:
            source_diac_indices = [0] * len(source_diac_indices)
        elif masking_factor == 0.0:
            pass
        else:
            if len(before_masking):
                mask = numpy.random.choice(
                    range(0, len(before_masking)),
                    replace=False,
                    size=int(float(len(before_masking) - 1) * masking_factor),
                )
                for idx in mask:
                    source_diac_indices[before_masking[idx]] = 0
        print("length", len(source_indices))
        assert len(source_indices) == (len(target_indices) and len(source_diac_indices))

        # here, we filter long sequence due to memory issues, in particular for self-attention models
        if len(source_indices) <= max_char_seq:
            hdf_source_letter_writer.insert_batch(
                numpy.asarray(
                    [
                        source_indices,
                    ],
                    dtype="int32",
                ),
                [len(source_indices)],
                seq_tag=["%s_%s_%i" % (sequence_name, str(args.masking_factor), i)],
            )
            hdf_target_writer.insert_batch(
                numpy.asarray(
                    [
                        target_indices,
                    ],
                    dtype="int32",
                ),
                [len(target_indices)],
                seq_tag=["%s_%s_%i" % (sequence_name, str(args.masking_factor), i)],
            )
            hdf_source_diacritic_writer.insert_batch(
                numpy.asarray(
                    [
                        source_diac_indices,
                    ],
                    dtype="int32",
                ),
                [len(source_diac_indices)],
                seq_tag=["%s_%s_%i" % (sequence_name, str(args.masking_factor), i)],
            )
        else:
            filtered_segments.write("%s_%s_%i" % (sequence_name, str(args.masking_factor), i) + "\n")

    hdf_source_letter_writer.close()
    hdf_target_writer.close()
    hdf_source_diacritic_writer.close()
    filtered_segments.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
