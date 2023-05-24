import numpy
import argparse
import sys
import time
from .dictionary import (
    arabic_index_dict,
    harakat,
    harakat_index_dict,
)

from returnn.datasets.generating import StaticDataset
from returnn import rnn


class ReturnnDiacritizer:
    """
    This is a helper class for wrapping the RETURNN functions needed to run diacritization given a model config
    """

    def __init__(self, config, epoch=None, model_checkpoint=None, device="cpu", search=False):
        """

        :param str config: path to a RETURNN model config
        :param int|None epoch: int determines which checkpoint to use
        :param str|None model_checkpoint: load from a specific checkpoint (can point to .index or .meta)
        :param str device: "cpu" or "gpu"
        :param bool search: if set True, autoregressive left-to-right search gets activated.
        """
        self.search = search
        print("device: %s" % device, file=sys.stderr)
        returnn_params = ["++log_verbosity", "5", "++device", device]
        if search:
            returnn_params += ["++task", "search"]
        if epoch:
            returnn_params += ["++load_epoch", str(epoch)]
        if model_checkpoint:
            returnn_params += ["++load", str(model_checkpoint)]
        rnn.init(config, returnn_params)
        rnn.engine.init_network_from_config(rnn.config)
        self.rnn_output = {"output": rnn.engine.network.get_default_output_layer().output.placeholder}

        self.inverse_harakat_dict = {0: "_"}
        for i, char in enumerate(harakat):
            self.inverse_harakat_dict[i + 1] = char

    def diacritze_line(self, line, twoSDiac=False, masking_factor=1.0):
        index_sequence = []
        index_sequence_diac = []
        line = line.strip()
        if twoSDiac:
            for char in line:
                if char in harakat:
                    if len(index_sequence_diac) == 0:
                        # sentence start with an harakat, skip
                        continue
                    if index_sequence_diac[-1] != 0:
                        # double harakat with shadda in front
                        index_sequence_diac[-1] = len(harakat) + harakat_index_dict[char]
                    else:
                        index_sequence_diac[-1] = harakat_index_dict[char]
                else:
                    index_sequence.append(arabic_index_dict.get(char, 0))
                    index_sequence_diac.append(0)

            # randomly select a percent of a list and set it to zero
            source_diac_indices = index_sequence_diac.copy()
            before_masking = [p for p, e in enumerate(source_diac_indices) if e != 0]

            # Due to the mask size
            if masking_factor == 1.0:
                source_diac_indices = [0] * len(source_diac_indices)
            elif masking_factor == 0.0:
                source_diac_indices = source_diac_indices
            else:
                if len(before_masking):
                    mask = numpy.random.choice(
                        range(0, len(before_masking)),
                        replace=False,
                        size=int(float(len(before_masking) - 1) * masking_factor),
                    )
                    for idx in mask:
                        source_diac_indices[before_masking[idx]] = 0

            data = numpy.asarray(index_sequence, dtype="int32")
            diacritic = numpy.asarray(source_diac_indices, dtype="int32")
            feed_dict = {"data": data, "diacritic": diacritic}

            static_dataset = StaticDataset([feed_dict], output_dim={"data": (71, 1), "diacritic": (17, 1)})
        else:
            index_sequence = [arabic_index_dict.get(char, 0) for char in line]
            data = numpy.asarray(index_sequence, dtype="int32")
            feed_dict = {"data": data}
            static_dataset = StaticDataset([feed_dict], output_dim={"data": (71, 1)})

        line_no_harakat = "".join([char for char in line if char not in harakat])

        if not self.search:
            # when we have no search and only forward pass
            result = rnn.engine.run_single(static_dataset, 0, self.rnn_output)["output"]
            result = numpy.squeeze(numpy.argmax(result, axis=2))
        else:
            # when we have autoregressive model (left-to-right)
            result = rnn.engine.search_single(static_dataset, 0, "output")[0][1]

        output = ""

        assert len(line_no_harakat) == len(result)
        for char, harakat_idx in zip(line_no_harakat, result):
            output += char
            if harakat_idx != 0:
                if harakat_idx > len(harakat):
                    # double harakat with shadda
                    output += self.inverse_harakat_dict[7]
                    output += self.inverse_harakat_dict[harakat_idx - len(harakat)]
                else:
                    output += self.inverse_harakat_dict[harakat_idx]

        return output


def main():
    parser = argparse.ArgumentParser(description="run RETURNN diacritization")

    parser.add_argument("config", help="path to the RETURNN config")
    parser.add_argument("--load-epoch", type=int, help="model epoch to load")
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        type=str,
        help="load from a specific checkpoint",
    )
    parser.add_argument("--from-file", type=str, default=None, help="read from file instead of stdin")
    parser.add_argument("--to-file", type=str, default=None, help="write to file instead of stdout")
    parser.add_argument(
        "--twoSDiac",
        action="store_true",
        help="set it if you want to run 2SDiac diacritization model with partially diacritized input",
    )
    parser.add_argument(
        "--masking_factor",
        default=1.0,
        help="random mask to run twoSDiac diacritization model",
    )
    parser.add_argument("--search", action="store_true", help="if we run autoregressive search or not")
    parser.add_argument("--device", type=str, default="cpu", help="device to use, 'cpu' or 'gpu'")

    args = parser.parse_args()
    masking_factor = float(args.masking_factor)

    diacritizer = ReturnnDiacritizer(
        args.config,
        args.load_epoch,
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        search=args.search,
    )

    single_times = 0
    global_start_time = time.time()

    if args.from_file:
        in_stream = open(args.from_file, "rt")
    else:
        in_stream = sys.stdin

    output_file = None
    if args.to_file:
        output_file = open(args.to_file, "wt")

    for i, line in enumerate(in_stream):
        start_time = time.time()
        output = diacritizer.diacritze_line(line, args.twoSDiac, masking_factor)
        if output_file:
            output_file.write(output + "\n")
        else:
            print(output)
        total_time = time.time() - start_time
        single_times += total_time
    print(
        "Completed in %.2f, single calls took %.2f" % (time.time() - global_start_time, single_times),
        file=sys.stderr,
    )


if __name__ == "__main__":
    # execute only if run as a script
    main()
