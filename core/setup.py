from core.clause import Atom, Predicate
from core.rules import RuleTemplate, RulesManager, LanguageFrame, ProgramTemplate

ON = Predicate("on", 2)
TOP = Predicate("top", 1)
MOVE = Predicate("move", 2)
SAME = Predicate("same", 2)
FLOOR = Predicate("floor", 1)

invented1 = Predicate("invented1", 2)
invented2 = Predicate("invented2", 2)


def setup_blockworld(n_blocks=4):
    BK = set([]).union(
        {Atom(SAME, [str(i), str(i)]) for i in range(n_blocks+1)}, \
        {Atom(FLOOR, ['0'])}
    )
    constants = list(map(str, range(1, n_blocks + 1)))[:n_blocks] + ["0"]
    targets = [ON, TOP]
    extentional = [FLOOR, SAME, ON, TOP, MOVE]
    language = LanguageFrame(targets, extensional=extentional, constants=constants)

    invented_preds = [invented1, invented2]

    rule_temps = {
        invented1: [RuleTemplate(0, False), RuleTemplate(0, False)],
        invented2: [RuleTemplate(0, True), RuleTemplate(0, True)],
        ON: [RuleTemplate(2, True), RuleTemplate(2, True),
             RuleTemplate(2, True)],
        TOP: [RuleTemplate(2, True), RuleTemplate(2, True),
              RuleTemplate(2, True)],
    }
    prog_temp = ProgramTemplate(invented_preds, rule_temps, forward_n=1)

    man = RulesManager(n_blocks, language, prog_temp, save_dir="exps/block_world")
    return man, BK


def check():
    man4 = setup_blockworld(n_blocks=4)[0]
    man5 = setup_blockworld(n_blocks=5)[0]
    man6 = setup_blockworld(n_blocks=6)[0]
    man7 = setup_blockworld(n_blocks=7)[0]
    man8 = setup_blockworld(n_blocks=8)[0]
    man9 = setup_blockworld(n_blocks=9)[0]
    intensionals = list(man4.all_clauses.keys())
    intensionals2 = list(man5.all_clauses.keys())
    assert intensionals == intensionals2
    assert man4.all_grounds != man5.all_grounds != man6.all_grounds
    print('all grounds num when n_blocks=4: ', len(man4.all_grounds))
    print('all grounds num when n_blocks=5: ', len(man5.all_grounds))
    print('all grounds num when n_blocks=6: ', len(man6.all_grounds))
    print('all grounds num when n_blocks=7: ', len(man7.all_grounds))
    print('all grounds num when n_blocks=8: ', len(man8.all_grounds))
    print('all grounds num when n_blocks=9: ', len(man9.all_grounds))
    for intensional in intensionals:
        for i in range(len(man4.program_template.rule_temps[intensional])):
            rules4 = man4.all_clauses[intensional][i]
            rules5 = man5.all_clauses[intensional][i]
            rules6 = man6.all_clauses[intensional][i]
            rules7 = man7.all_clauses[intensional][i]
            rules8 = man8.all_clauses[intensional][i]
            rules9 = man9.all_clauses[intensional][i]
            assert rules4 == rules5 == rules6 == rules7 == rules8 == rules9


if __name__ == "__main__":
    check()