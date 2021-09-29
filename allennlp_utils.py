from copy import deepcopy
from allennlp.common.params import Params
from allennlp.predictors import Predictor
from allennlp_models.generation.predictors import Seq2SeqPredictor
from tqdm import tqdm
import numpy as np
import os


def load_predictor_from_model_name(model_name, serialization_dir, cuda_device, predictor_name='seq2seq'):
    from allennlp.models.archival import _load_dataset_readers, _load_model
    config = Params.from_file(os.path.join(serialization_dir, 'config.json'))
    dataset_reader, validation_dataset_reader = _load_dataset_readers(config.duplicate(), serialization_dir)
    model = _load_model(config.duplicate(),
                        os.path.join(serialization_dir, model_name),
                        serialization_dir,
                        cuda_device=cuda_device)
    predictor = Predictor.by_name(predictor_name)(model, validation_dataset_reader)
    return predictor


def seq2seq_predict_on_instances(predictor, instances, batch_size, with_target_tokens=False):
    instances_copy = deepcopy(instances)
    if not with_target_tokens:
        for instance in instances_copy:
            if 'target_tokens' in instance.fields:
                del instance.fields['target_tokens']
    outputs = []
    for idx in tqdm(range(0, len(instances), batch_size), total=len(instances) // batch_size):
        batch_output = predictor.predict_batch_instance(instances_copy[idx:idx + batch_size])
        outputs.extend(batch_output)
    return outputs


def seq2seq_predict_on_instances_by_model(model, instances, batch_size, with_target_tokens=False):
    if not with_target_tokens and 'target_tokens' in instances[0].fields:
        instances_copy = deepcopy(instances)
        for instance in instances_copy:
            del instance.fields['target_tokens']
    else:
        instances_copy = instances
    model.eval()
    outputs = []
    for idx in tqdm(range(0, len(instances), batch_size), total=len(instances) // batch_size):
        batch_output = model.forward_on_instances(instances_copy[idx:idx + batch_size])
        outputs.extend(batch_output)
    return outputs


def evaluate_on_instances_by_model(model, reader, instances, batch_size):
    model.eval()
    outputs = seq2seq_predict_on_instances_by_model(model, instances, batch_size)
    corr_cnt = 0
    for instance, output in zip(instances, outputs):
        tgt = ' '.join([
            x.text for x in instance.fields["target_tokens"].tokens
            if x not in [reader._start_token, reader._end_token]
        ])
        # tgt2 = instance.fields['meta_data']['next_ob']
        # assert tgt == tgt2
        pred = ' '.join(output['predicted_tokens'][0])
        if pred == tgt:
            corr_cnt += 1
    acc = corr_cnt / len(instances)
    return acc