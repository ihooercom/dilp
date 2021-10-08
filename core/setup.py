from core.clause import Atom, Predicate, str2clause
from core.rules import RuleTemplate, RulesManager, LanguageFrame, ProgramTemplate

ON = Predicate("on", 2)
NOT_ON = Predicate("not_on", 2)
ADD_ON = Predicate("add_on", 2)
DEL_ON = Predicate("del_on", 2)

TOP = Predicate("top", 1)
ADD_TOP = Predicate("add_top", 1)
DEL_TOP = Predicate("del_top", 1)

MOVE = Predicate("move", 2)
SAME = Predicate("same", 2)
NOT_SAME = Predicate("not_same", 2)
FLOOR = Predicate("floor", 1)
NOT_FLOOR = Predicate("not_floor", 1)

invented1 = Predicate("invented1", 2)
invented2 = Predicate("invented2", 2)
invented3 = Predicate("invented3", 2)
invented4 = Predicate("invented4", 2)
invented5 = Predicate("invented5", 2)
invented6 = Predicate("invented6", 2)
invented7 = Predicate("invented7", 2)
invented8 = Predicate("invented8", 2)
invented9 = Predicate("invented9", 2)

gt_rules = {
    invented1: [str2clause("invented1(X,Y):-move(X,Y),not_same(X,Y)")],
    invented2: [str2clause("invented2(X,Y):-floor(Y),not_on(X,Y)")],
    invented3: [str2clause("invented3(X,Y):-not_floor(Y),on(X,Y)")],
    invented4: [str2clause("invented4(X,Y):-invented1(X,Y),top(X)")],
    invented5:
    [str2clause("invented5(X,Y):-invented4(X,Y),top(Y)"),
     str2clause("invented5(X,Y):-invented2(X,Y),invented4(X,Y)")],
    ADD_ON: [str2clause("add_on(X,Y):-invented5(X,Y),invented5(X,Y)")],
    DEL_ON: [str2clause("del_on(X,Y):-invented5(X,Z),on(X,Y)")],
    ADD_TOP: [str2clause("add_top(X):-invented3(Y,X),invented5(Y,Z)")],
    DEL_TOP: [str2clause("del_top(X):-invented5(Y,X),top(X)")],
}


def setup_blockworld(n_blocks=4):
    BK = set([]).union(
        {Atom(NOT_SAME, [str(i), str(j)]) for i in range(n_blocks+1) for j in range(n_blocks+1) if i!=j}, \
        {Atom(FLOOR, ['0'])},
        {Atom(NOT_FLOOR, [str(i)]) for i in range(1, n_blocks+1)},
    )
    constants = list(map(str, range(1, n_blocks + 1)))[:n_blocks] + ["0"]
    ALL_ON_ATOMS = [Atom(ON, [str(i), str(j)]) for i in range(0, n_blocks + 1) for j in range(0, n_blocks + 1)]
    ALL_TOP_ATOMS = [Atom(TOP, [str(i)]) for i in range(0, n_blocks + 1)]
    targets = [ADD_ON, DEL_ON, ADD_TOP, DEL_TOP]
    extentional = [FLOOR, NOT_FLOOR, NOT_SAME, ON, NOT_ON, TOP, MOVE]
    language = LanguageFrame(targets, extensional=extentional, constants=constants)

    invented_preds = [invented1, invented2, invented3, invented4, invented5, invented6, invented7]

    rule_temps = {
        invented1: [RuleTemplate(0, False), RuleTemplate(0, False)],
        invented2: [RuleTemplate(0, False), RuleTemplate(0, False)],
        invented3: [RuleTemplate(0, False), RuleTemplate(0, False)],
        invented4: [RuleTemplate(0, True), RuleTemplate(0, True)],
        invented5: [RuleTemplate(0, True), RuleTemplate(0, True)],
        invented6: [RuleTemplate(0, True), RuleTemplate(1, True)],
        invented7: [RuleTemplate(0, True), RuleTemplate(1, True)],
        ADD_ON: [RuleTemplate(2, True)],
        DEL_ON: [RuleTemplate(2, True)],
        ADD_TOP: [RuleTemplate(2, True)],
        DEL_TOP: [RuleTemplate(2, True)],
    }
    prog_temp = ProgramTemplate(invented_preds, rule_temps, forward_n=1)

    man = RulesManager(n_blocks, language, prog_temp, save_dir="exps/block_world")
    return man, BK, targets, ALL_ON_ATOMS, ALL_TOP_ATOMS


def check():
    mans = []
    n_blocks_list = [4, 5]
    for n_blocks in n_blocks_list:
        mans.append(setup_blockworld(n_blocks=n_blocks)[0])
    intensionals = list(mans[0].all_clauses.keys())
    intensionals2 = list(mans[1].all_clauses.keys())
    assert intensionals == intensionals2
    for i, n_blocks in enumerate(n_blocks_list):
        print(f'all grounds num when n_blocks={n_blocks}: ', len(mans[i].all_grounds))
    for intensional in intensionals:
        for i in range(len(mans[0].program_template.rule_temps[intensional])):
            rules_list = [mans[j].all_clauses[intensional][i] for j in range(len(n_blocks_list))]
            print(intensional.name, "rules num: ", len(rules_list[0]))
            for j in range(1, len(n_blocks_list)):
                assert rules_list[0] == rules_list[j]


if __name__ == "__main__":
    check()