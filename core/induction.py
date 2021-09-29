from typing import Dict, Any
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import os
import json

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TensorField, MetadataField
from allennlp.models.model import Model
from allennlp.training.callbacks import TrainerCallback
from allennlp.training.metrics.average import Average

from core.clause import str2atom
from core.setup import setup_blockworld, ON, TOP, invented1, invented2
from copy import deepcopy


def str2atoms(atoms_str):
    if not atoms_str:
        return []
    atoms = list(map(str2atom, sorted(atoms_str.split())))
    return atoms


@DatasetReader.register('transition')
class TransitionDatasetReader(DatasetReader):
    def __init__(self, n_blocks=4, **kwargs):
        super().__init__(**kwargs)
        rules_man, BK = setup_blockworld(n_blocks)
        self.ground_atoms = rules_man.all_grounds
        self.BK = BK

    def _read(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            ob = item['ob']
            ac = item['ac']
            next_ob = item['next_ob']
            yield self.text_to_instance(ob, ac, next_ob, meta_data={})

    def text_to_instance(self, ob, ac, next_ob, meta_data={}) -> Instance:
        ob_atoms, ac_atom, next_ob_atoms = [str2atoms(x) for x in [ob[0], ac, next_ob[0]]]
        input_val = self.axioms2valuation(list(self.BK) + ob_atoms + ac_atom)
        output_val = self.axioms2valuation(next_ob_atoms)
        meta_data_dict = deepcopy(meta_data)
        meta_data_dict['next_ob'] = next_ob
        fields = {
            'input_valuation': TensorField(input_val),
            'output_valuation': TensorField(output_val),
            "meta_data": MetadataField(meta_data_dict)
        }
        return Instance(fields)

    def axioms2valuation(self, axioms):
        '''
        :param axioms: list of Atoms, background knowledge
        :return: a valuation vector
        '''
        result = np.zeros(len(self.ground_atoms), dtype=np.float32)
        for i, atom in enumerate(self.ground_atoms):
            if atom in axioms:
                result[i] = 1.0
        return result

    def apply_token_indexers(self, instance: Instance) -> None:
        pass


@TrainerCallback.register('my')
class MyCallbak(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rules_reulst = {"training": []}

    def on_epoch(
        self,
        trainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        epoch_data_dict = trainer.model.get_rules_definition(threshold=0.1)
        self.rules_reulst['training'].append({"epoch": epoch, "rules": epoch_data_dict})

    def on_end(
        self,
        trainer,
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        path = os.path.join(self.serialization_dir, 'rules_result.json')
        with open(path, 'w') as f:
            json.dump(self.rules_reulst, f, indent=4)


@Model.register('dilp')
class DILP(Model):
    def __init__(self, n_blocks=4, **kwargs):
        super().__init__(**kwargs)
        self.init(n_blocks)
        self.rule_weights = OrderedDict()
        self.__init__rule_weights()
        self.metric = Average()

    def init(self, n_blocks):
        self.rules_man, _ = setup_blockworld(n_blocks)
        self.ground_atoms = self.rules_man.all_grounds
        self.output_weight = self.axioms2valuation([a for a in self.ground_atoms if a.predicate in [ON, TOP]])
        self.criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(self.output_weight))

    def __init__rule_weights(self):
        for predicate, clauses in self.rules_man.all_clauses.items():
            params_list = []
            for i in range(len(clauses)):
                weight = Parameter(torch.Tensor(len(clauses[i])))
                weight.data.normal_(0, 1)
                params_list.append(weight)
            pname = f'rule_weights_{predicate.name}'
            setattr(self, pname, nn.ParameterList(params_list))
            self.rule_weights[predicate] = getattr(self, pname)

    def get_rules_definition(self, threshold=0.1):
        data = {}
        for predicate in self.rules_man.all_clauses:
            data[predicate.name] = []
            result = self.get_predicate_definition(predicate)
            for weight, clause in result:
                if weight > threshold:
                    data[predicate.name].append(str(round(weight, 3)) + ': ' + str(clause))
        return data

    def get_predicates_definition(self, threshold=0.0):
        result = {}
        for predicate in self.rules_man.all_clauses:
            result[predicate] = self.get_predicate_definition(predicate, threshold)
        return result

    def get_predicate_definition(self, predicate, threshold=0.0):
        clauses = self.rules_man.all_clauses[predicate]
        rules_weights = self.rule_weights[predicate]
        result = []
        for i, rule_weights in enumerate(rules_weights):
            weights = torch.nn.functional.softmax(rule_weights)
            indexes = torch.nonzero(weights > threshold)[:, 0]
            for j in range(len(indexes)):
                result.append((weights[indexes[j]].item(), str(clauses[i][indexes[j]])))
        return result

    def axioms2valuation(self, axioms):
        '''
        :param axioms: list of Atoms, background knowledge
        :return: a valuation vector
        '''
        result = np.zeros(len(self.ground_atoms), dtype=np.float32)
        for i, atom in enumerate(self.ground_atoms):
            if atom in axioms:
                result[i] = 1.0
        return result

    def valuation2atoms(self, valuation, threshold=0.5):
        result = OrderedDict()
        for i, value in enumerate(valuation):
            if value >= threshold:
                result[self.ground_atoms[i]] = float(value)
        return result

    def forward(self, input_valuation, output_valuation=None, meta_data=None):
        valuation = input_valuation
        for _ in range(self.rules_man.program_template.forward_n):
            valuation = self.inference_step(input_valuation, valuation)

        # pred_atoms_with_scores = [self.valuation2atoms(x, 0.5) for x in valuation.detach().cpu().numpy()]
        # print(list(map(lambda x: (str(x[0]), x[1]), pred_atoms_with_scores[0].items())))

        output_dict = {}
        if output_valuation is not None:
            loss = self.criterion(valuation, output_valuation)
            pred = [' '.join(sorted(list(map(str, self.valuation2atoms(x, 0.5))))) for x in valuation.detach()]
            tgt = [x['next_ob'][0] for x in meta_data]
            acc = np.sum(np.array(pred) == np.array(tgt)) / len(pred)
            self.metric(acc)
            output_dict['loss'] = loss
        output_dict['predictions'] = valuation.detach()
        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valuation = output_dict["predictions"]
        prediction_with_scores = [{str(k): v for k, v in self.valuation2atoms(x, 0.5).items()} for x in valuation]
        output_dict['predicted_tokens'] = prediction_with_scores
        return output_dict

    # def inference_step(self, input_valuation, valuation, step):
    #     deduced_valuation = torch.zeros_like(valuation)
    #     for predicate, matrix in self.rules_man.deduction_matrices.items():
    #         deduced_valuation += self.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
    #     return deduced_valuation + input_valuation

    def inference_step(self, input_valuation, valuation):
        deduced_valuation = torch.zeros_like(valuation)
        invented1_valuation = self.inference_single_predicate(input_valuation,
                                                              self.rules_man.deduction_matrices[invented1],
                                                              self.rule_weights[invented1])
        valuation = invented1_valuation + input_valuation
        invented2_valuation = self.inference_single_predicate(valuation, self.rules_man.deduction_matrices[invented2],
                                                              self.rule_weights[invented2])
        valuation = invented2_valuation + invented1_valuation + input_valuation
        for predicate, matrix in self.rules_man.deduction_matrices.items():
            if predicate in [ON, TOP]:
                deduced_valuation += self.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
        return deduced_valuation

    def inference_single_predicate(self, valuation, deduction_matrices, rule_weights):
        '''
        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = [[] for _ in rule_weights]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(self.inference_single_clause(valuation, matrix))

        c_p = None
        for i in range(len(result_valuations)):
            valuations = torch.stack(result_valuations[i], dim=-1)
            prob_rule_weights = torch.nn.functional.softmax(rule_weights[i])[None, None, :]
            if c_p == None:
                c_p = torch.sum(prob_rule_weights * valuations, dim=-1)
            else:
                c_p = prob_sum(c_p, torch.sum(prob_rule_weights * valuations, dim=-1))
        return c_p

    def inference_single_clause(self, valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        X1 = X[:, :, 0, None]
        X2 = X[:, :, 1, None]
        Y1 = torch.index_select(valuation, dim=1,
                                index=torch.LongTensor(X1).view(-1).to(valuation.device)).view(-1, *X1.shape[:2])
        Y2 = torch.index_select(valuation, dim=1,
                                index=torch.LongTensor(X2).view(-1).to(valuation.device)).view(-1, *X2.shape[:2])
        Z = Y1 * Y2
        return torch.max(Z, dim=2)[0]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"acc": self.metric.get_metric(reset=reset)}
        return metrics


def prob_sum(x, y):
    return x + y - x * y
