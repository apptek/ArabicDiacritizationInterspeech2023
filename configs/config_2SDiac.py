#!rnn.py

batch_size = 10000
device = "gpu"
learning_rate = 0.001
learning_rate_control = "newbob_multi_epoch"
learning_rate_control_min_num_epochs_per_new_lr = 3
learning_rate_control_relative_error_relative_lr = True
learning_rate_file = "learning_rates"
newbob_learning_rate_decay = 0.9
newbob_multi_num_epochs = 3
newbob_multi_update_interval = 1
num_epochs = 50
optimizer = {"class": "Adam"}
save_interval = 1
target = "classes"
task = "train"
tf_log_memory_usage = True
log = ["./returnn.log"]
log_batch_size = True
log_verbosity = 5
max_seqs = 300
model = "../models/2SDiac/epoch"
use_tensorflow = True


extern_data = {
    "classes": {
        "available_for_inference": False,
        "dim": 17,
        "shape": (None,),
        "sparse": True,
    },
    "data": {
        "available_for_inference": True,
        "dim": 71,
        "shape": (None,),
        "sparse": True,
    },
    "diacritic": {
        "available_for_inference": True,
        "dim": 17,
        "shape": (None,),
        "sparse": True,
    },
}

network = {
    "decision": {
        "class": "decide",
        "from": ["output"],
        "loss": "edit_distance",
        "target": "classes",
    },
    "embed_layer": {
        "class": "combine",
        "from": ["source_embed_dropout", "source_embed_dropout_diac"],
        "kind": "add",
    },
    "enc_01": {"class": "copy", "from": ["enc_01_ff_out"]},
    "enc_01_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_01_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_01_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_01_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_01_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_01_ff_conv2"]},
    "enc_01_ff_laynorm": {"class": "layer_norm", "from": ["enc_01_self_att_out"]},
    "enc_01_ff_out": {
        "class": "combine",
        "from": ["enc_01_self_att_out", "enc_01_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_01_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_01_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_01_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_01_self_att_lin"],
    },
    "enc_01_self_att_laynorm": {"class": "layer_norm", "from": ["embed_layer"]},
    "enc_01_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_01_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_01_self_att_out": {
        "class": "combine",
        "from": ["embed_layer", "enc_01_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_02": {"class": "copy", "from": ["enc_02_ff_out"]},
    "enc_02_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_02_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_02_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_02_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_02_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_02_ff_conv2"]},
    "enc_02_ff_laynorm": {"class": "layer_norm", "from": ["enc_02_self_att_out"]},
    "enc_02_ff_out": {
        "class": "combine",
        "from": ["enc_02_self_att_out", "enc_02_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_02_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_02_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_02_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_02_self_att_lin"],
    },
    "enc_02_self_att_laynorm": {"class": "layer_norm", "from": ["enc_01"]},
    "enc_02_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_02_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_02_self_att_out": {
        "class": "combine",
        "from": ["enc_01", "enc_02_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_03": {"class": "copy", "from": ["enc_03_ff_out"]},
    "enc_03_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_03_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_03_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_03_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_03_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_03_ff_conv2"]},
    "enc_03_ff_laynorm": {"class": "layer_norm", "from": ["enc_03_self_att_out"]},
    "enc_03_ff_out": {
        "class": "combine",
        "from": ["enc_03_self_att_out", "enc_03_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_03_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_03_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_03_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_03_self_att_lin"],
    },
    "enc_03_self_att_laynorm": {"class": "layer_norm", "from": ["enc_02"]},
    "enc_03_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_03_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_03_self_att_out": {
        "class": "combine",
        "from": ["enc_02", "enc_03_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_04": {"class": "copy", "from": ["enc_04_ff_out"]},
    "enc_04_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_04_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_04_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_04_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_04_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_04_ff_conv2"]},
    "enc_04_ff_laynorm": {"class": "layer_norm", "from": ["enc_04_self_att_out"]},
    "enc_04_ff_out": {
        "class": "combine",
        "from": ["enc_04_self_att_out", "enc_04_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_04_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_04_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_04_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_04_self_att_lin"],
    },
    "enc_04_self_att_laynorm": {"class": "layer_norm", "from": ["enc_03"]},
    "enc_04_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_04_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_04_self_att_out": {
        "class": "combine",
        "from": ["enc_03", "enc_04_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_05": {"class": "copy", "from": ["enc_05_ff_out"]},
    "enc_05_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_05_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_05_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_05_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_05_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_05_ff_conv2"]},
    "enc_05_ff_laynorm": {"class": "layer_norm", "from": ["enc_05_self_att_out"]},
    "enc_05_ff_out": {
        "class": "combine",
        "from": ["enc_05_self_att_out", "enc_05_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_05_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_05_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_05_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_05_self_att_lin"],
    },
    "enc_05_self_att_laynorm": {"class": "layer_norm", "from": ["enc_04"]},
    "enc_05_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_05_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_05_self_att_out": {
        "class": "combine",
        "from": ["enc_04", "enc_05_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_06": {"class": "copy", "from": ["enc_06_ff_out"]},
    "enc_06_ff_conv1": {
        "activation": "relu",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_06_ff_laynorm"],
        "n_out": 1024,
        "with_bias": True,
    },
    "enc_06_ff_conv2": {
        "activation": None,
        "class": "linear",
        "dropout": 0.1,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_06_ff_conv1"],
        "n_out": 256,
        "with_bias": True,
    },
    "enc_06_ff_drop": {"class": "dropout", "dropout": 0.1, "from": ["enc_06_ff_conv2"]},
    "enc_06_ff_laynorm": {"class": "layer_norm", "from": ["enc_06_self_att_out"]},
    "enc_06_ff_out": {
        "class": "combine",
        "from": ["enc_06_self_att_out", "enc_06_ff_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "enc_06_self_att_att": {
        "attention_dropout": 0.1,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_06_self_att_laynorm"],
        "n_out": 256,
        "num_heads": 4,
        "total_key_dim": 256,
    },
    "enc_06_self_att_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["enc_06_self_att_lin"],
    },
    "enc_06_self_att_laynorm": {"class": "layer_norm", "from": ["enc_05"]},
    "enc_06_self_att_lin": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_06_self_att_att"],
        "n_out": 256,
        "with_bias": False,
    },
    "enc_06_self_att_out": {
        "class": "combine",
        "from": ["enc_05", "enc_06_self_att_drop"],
        "kind": "add",
        "n_out": 256,
    },
    "encoder": {"class": "layer_norm", "from": ["enc_06"]},
    "output": {
        "class": "rec",
        "from": ["encoder"],
        "max_seq_len": "max_len_from('base:encoder') * 1",
        "target": "classes",
        "unit": {
            "output": {
                "beam_size": 5,
                "class": "choice",
                "from": ["output_prob"],
                "initial_output": 0,
                "target": "classes",
            },
            "output0_embed": {
                "activation": None,
                "class": "linear",
                "forward_weights_init":"variance_scaling_initializer(mode='fan_in',distribution='uniform', scale=0.78)",
                "from": ["output"],
                "initial_output": 0,
                "n_out": 256,
                "with_bias": False,
            },
            "output_embed": {
                "class": "combine",
                "from": ["output0_embed"],
                "kind": "add",
            },
            "output_embed_dropout": {
                "class": "dropout",
                "dropout": 0.1,
                "from": ["output_embed_with_pos"],
            },
            "output_embed_weighted": {
                "class": "eval",
                "eval": "source(0) * 16.000000",
                "from": ["prev:output_embed"],
            },
            "output_embed_with_pos": {
                "add_to_input": True,
                "class": "positional_encoding",
                "from": ["output_embed_weighted"],
            },
            "output_prob": {
                "class": "softmax",
                "dropout": 0.3,
                "from": ["readout"],
                "loss": "ce",
                "loss_opts": {"label_smoothing": 0.1},
                "target": "classes",
            },
            "readout": {
                "activation": None,
                "class": "linear",
                "from": ["prev:output_embed_dropout", "data:source"],
                "n_out": 256,
            },
        },
    },
    "source0_embed": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["data"],
        "n_out": 256,
        "with_bias": False,
    },
    "source0_embed_diac": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["data:diacritic"],
        "n_out": 256,
        "with_bias": False,
    },
    "source_embed": {"class": "combine", "from": ["source0_embed"], "kind": "add"},
    "source_embed_diac": {
        "class": "combine",
        "from": ["source0_embed_diac"],
        "kind": "add",
    },
    "source_embed_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["source_embed_with_pos"],
    },
    "source_embed_dropout_diac": {
        "class": "dropout",
        "dropout": 0.1,
        "from": ["source_embed_with_pos_diac"],
    },
    "source_embed_weighted": {
        "class": "eval",
        "eval": "source(0) * 16.000000",
        "from": ["source_embed"],
    },
    "source_embed_weighted_diac": {
        "class": "eval",
        "eval": "source(0) * 16.000000",
        "from": ["source_embed_diac"],
    },
    "source_embed_with_pos": {
        "add_to_input": True,
        "class": "positional_encoding",
        "from": ["source_embed_weighted"],
    },
    "source_embed_with_pos_diac": {
        "add_to_input": True,
        "class": "positional_encoding",
        "from": ["source_embed_weighted_diac"],
    },
}

train = {
    "class": "MetaDataset",
    "data_dims": {"classes": (17, 1), "data": (71, 1), "diacritic": (17, 1)},
    "data_map": {
        "classes": ("target", "data"),
        "data": ("source", "data"),
        "diacritic": ("source_diac", "data"),
    },
    "datasets": {
        "source": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf training files
            ],
            "seq_ordering": "random",
        },
        "source_diac": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf training files
            ],
        },
        "target": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf training files
            ],
        },
    },
    "seq_order_control_dataset": "source",
}

dev = {
    "class": "MetaDataset",
    "data_dims": {"classes": (17, 1), "data": (71, 1), "diacritic": (17, 1)},
    "data_map": {
        "classes": ("target", "data"),
        "data": ("source", "data"),
        "diacritic": ("source_diac", "data"),
    },
    "datasets": {
        "source": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf dev files
            ],
            "seq_ordering": "random",
        },
        "source_diac": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf dev files
            ],
        },
        "target": {
            "class": "HDFDataset",
            "files": [
                # paths to hdf dev files
            ],
        },
    },
    "seq_order_control_dataset": "source",
}

from TFUtil import DimensionTag

input_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="input time")
extern_data["data"]["same_dim_tags_as"] = {"t": input_time}
extern_data["diacritic"]["same_dim_tags_as"] = {"t": input_time}
extern_data["classes"]["same_dim_tags_as"] = {"t": input_time}
search_output_layer = "decision"
