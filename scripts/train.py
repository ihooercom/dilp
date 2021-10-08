from allennlp.common import Params
from allennlp.commands.train import train_model
from core.induction import TransitionDatasetReader, DILP, MyCallbak

cfg = {
    "train_data_path": "data/block_world_n4/train.json",
    "validation_data_path": "data/block_world_n4/val.json",
    "dataset_reader": {
        "type": "transition",
        "n_blocks": 4,
        # "max_instances": 64
    },
    "validation_dataset_reader": {
        "type": "transition",
        "n_blocks": 4,
        # "max_instances": 64
    },
    "model": {
        "type": "dilp",
        # "regularizer": {
        #     "regexes": [[".*", {
        #         "type": "l1",
        #         "alpha": 1e-4
        #     }]]
        # },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8
        },
        # "batches_per_epoch": 5
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8
        },
    },
    "trainer": {
        "num_epochs":
        200,
        "cuda_device":
        1,
        "optimizer": {
            "type": "adam",
            "lr": 0.05,
        },
        "learning_rate_scheduler": {
            # "type": "constant",
            "type": "cosine",
            "t_initial": 5,
            "t_mul": 1.0,
            "eta_min": 0.01,
            "eta_mul": 1.0,
        },
        "validation_metric":
        "+acc",
        "callbacks": [{
            "type": "tensorboard",
            "summary_interval": 1,
            "should_log_parameter_statistics": False,
            "should_log_learning_rate": True
        }, {
            "type": "my"
        }],
        "checkpointer": {
            "keep_most_recent_by_count": 20
        },
        "random_seed":
        0,
        "numpy_seed":
        0,
        "pytorch_seed":
        0
    },
}

if __name__ == "__main__":
    config = Params(cfg)
    serialization_dir = 'exps/block_world/pt'
    train_model(config, serialization_dir=serialization_dir, force=False, recover=True)