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
            "batch_size": 64
        },
        # "batches_per_epoch": 10
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64
        },
    },
    "trainer": {
        "num_epochs":
        20,
        "cuda_device":
        1,
        "optimizer": {
            "type": "rmsprop",
            "lr": 0.1
        },
        "learning_rate_scheduler": {
            "type": "constant"
        },
        "validation_metric":
        "+acc",
        "callbacks": [{
            "type": "tensorboard",
            "summary_interval": 10,
            "should_log_parameter_statistics": False,
            "should_log_learning_rate": True
        }, {
            "type": "my"
        }],
        "checkpointer": {
            "keep_most_recent_by_count": 10
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
    train_model(config, serialization_dir=serialization_dir, force=False)