from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance
from allennlp.data import DatasetReader
from core.clause import str2atom, Atom
from core.setup import setup_blockworld, NOT_ON
from tqdm import tqdm
import numpy as np
import json

rules_man, BK, targets, ALL_ON_ATOMS, ALL_TOP_ATOMS = setup_blockworld(n_blocks=4)


def str2atoms(atoms_str):
    if not atoms_str:
        return []
    atoms = list(map(str2atom, sorted(atoms_str.split())))
    return atoms


def generate_next_ob_by_rule(ob, ac, rule) -> Instance:
    ob_atoms, ac_atom = [str2atoms(x) for x in [ob[0], ac]]
    not_on_atoms = [Atom(NOT_ON, a.terms) for a in ALL_ON_ATOMS if a not in ob_atoms]
    true_grounds = set(list(BK) + ob_atoms + not_on_atoms + ac_atom)
    deduced_true_grounds = rules_man.deduce_true_grounds_by_rule(rule, true_grounds)
    ob_atoms = set(ob_atoms)
    for a in deduced_true_grounds:
        if a.predicate.name.startswith('add_'):
            add_a = str2atom(str(a)[4:])
            ob_atoms.add(add_a)
        elif a.predicate.name.startswith('del_'):
            del_a = str2atom(str(a)[4:])
            if del_a in ob_atoms:
                ob_atoms.remove(del_a)
    next_ob = [' '.join([str(x) for x in sorted(list(map(str, ob_atoms)))])]
    return next_ob


def check_generated_rules_from_template():
    rule_str = '\n'.join([
        "invented1(X,Y):-move(X,Y),not_same(X,Y)",
        "invented2(X,Y):-floor(Y),not_on(X,Y)",
        "invented3(X,Y):-not_floor(Y),on(X,Y)",
        "invented4(X,Y):-invented1(X,Y),top(X)",
        "invented5(X,Y):-invented4(X,Y),top(Y)",
        "invented5(X,Y):-invented2(X,Y),invented4(X,Y)",
        "add_on(X,Y):-invented5(X,Y),invented5(X,Y)",
        "del_on(X,Y):-invented5(X,Z),on(X,Y)",
        "add_top(X):-invented3(Y,X),invented5(Y,Z)",
        "del_top(X):-invented5(Y,X),top(X)",
    ])
    file_path = 'data/block_world_n4/data.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    from tqdm import tqdm
    instances = data
    for instance in tqdm(instances, total=len(instances)):
        ob = instance['ob']
        ac = instance['ac']
        next_ob = instance['next_ob']
        rule = rules_man.str2rule(rule_str)
        print('*' * 100)
        print(ob)
        print(ac)
        print(next_ob)
        next_ob_gen = generate_next_ob_by_rule(ob, ac, rule)
        print(next_ob_gen)
        assert next_ob == next_ob_gen


class TmpDatasetReader(DatasetReader):
    def __init__(self, **kwargs):
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)
        file_path = 'data/block_world_n4/data.json'
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        with open('exps/block_world/candidate_rules_from_dilp.json', 'r') as f:
            self.cand_rules = json.load(f)

    def _read(self, file_path):
        instances = self.data
        for idx, r in self.shard_iterable(enumerate(self.cand_rules)):
            print(f'evaluate {idx}th rule...')

            corr = 0
            for instance in tqdm(instances, total=len(instances)):
                ob = instance['ob']
                ac = instance['ac']
                next_ob = instance['next_ob']
                rule_str = '\n'.join(r)
                rule = rules_man.str2rule(rule_str)
                # print('*' * 100)
                # print(ob)
                # print(ac)
                # print(next_ob)
                next_ob_gen = generate_next_ob_by_rule(ob, ac, rule)
                # print(next_ob_gen)
                if next_ob == next_ob_gen:
                    corr += 1
            acc = corr / len(instances)
            print(acc)
            yield self.text_to_instance(r, acc)

    def text_to_instance(self, rule, acc):
        meta_data = {'rule': rule, 'acc': acc}
        return Instance({'meta_data': MetadataField(meta_data)})


if __name__ == "__main__":
    # check_generated_rules_from_template()
    reader = TmpDatasetReader()
    data_loader = MultiProcessDataLoader(reader, "", batch_size=1, num_workers=12)
    instances = list(data_loader.iter_instances())
    stats = [inst.fields['meta_data'] for inst in instances]
    stats = [{'rule': x['rule'], 'acc': float(x['acc'])} for x in stats]
    stats = sorted(stats, key=lambda x: x['acc'], reverse=True)
    with open('exps/block_world/stats_of_candidate_rules.json', 'w') as f:
        json.dump(stats, f, indent=4)