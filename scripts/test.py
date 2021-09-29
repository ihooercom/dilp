import numpy as np
from copy import deepcopy
import random
import json
import torch
import os
from tqdm import tqdm
from allennlp.common import Params
from allennlp.predictors import Predictor
from core.induction import TransitionDatasetReader, DILP, MyCallbak
from allennlp_utils import load_predictor_from_model_name, evaluate_on_instances_by_model


def seq2seq_predict_on_instances_by_model(model, instances, batch_size, with_target=False):
    if not with_target and 'output_valuation' in instances[0].fields:
        instances_copy = deepcopy(instances)
        for instance in instances_copy:
            del instance.fields['output_valuation']
    else:
        instances_copy = instances
    model.eval()
    outputs = []
    for idx in tqdm(range(0, len(instances), batch_size), total=len(instances) // batch_size):
        batch_output = model.forward_on_instances(instances_copy[idx:idx + batch_size])
        outputs.extend(batch_output)
    return outputs


def compute_metric(instances, outputs):
    corr_cnt = 0
    for instance, output in zip(instances, outputs):
        tgt = instance.fields['meta_data']['next_ob'][0]
        pred = ' '.join(sorted(list(output['predicted_tokens'].keys())))
        if pred == tgt:
            corr_cnt += 1
        else:
            print('*' * 100)
            print(tgt)
            print(pred)
    acc = corr_cnt / len(instances)
    return acc


if __name__ == "__main__":
    validation_data_path = 'data/block_world_n4/val.json'
    serialization_dir = 'exps/block_world/pt'
    predictor = load_predictor_from_model_name("model_state_e20_b0.th", serialization_dir, cuda_device=0)
    predictor._dataset_reader.max_instances = 64

    instances = list(predictor._dataset_reader.read(validation_data_path))
    outputs = seq2seq_predict_on_instances_by_model(predictor._model, instances, batch_size=64)
    acc = compute_metric(instances, outputs)
    print('acc: ', acc)
