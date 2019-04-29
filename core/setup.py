from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
import ray
from core.rules import *
from core.induction import *
from core.rl import *
from core.clause import str2atom,str2clause
from core.NTP import NeuralProver, RLProver, SymbolicNeuralProver
from core.symbolicEnvironment import *

def setup_predecessor():
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    positive = [Atom(Predicate("predecessor", 2), [constants[i], constants[i+2]]) for i in range(8)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(10) for j in range(10)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, False), RuleTemplate(0, False)]},
                                   4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_fizz():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    fizz = Predicate("fizz", 1)
    pred1 = Predicate("pred1", 2)
    pred2 = Predicate("pred2", 2)

    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(fizz, [constants[i]]) for i in range(0, 10, 3)]
    all_atom = [Atom(fizz, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(fizz, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred1, pred2], {fizz: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                                    pred1: [RuleTemplate(1, True),],
                                                    pred2: [RuleTemplate(1, True),],},
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_even():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    target = Predicate("even", 1)
    pred = Predicate("pred", 2)
    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(target, [constants[i]]) for i in range(0, 10, 2)]
    all_atom = [Atom(target, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(target, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred], {target: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                            pred: [RuleTemplate(1, True),RuleTemplate(1, False)],
                                            },
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_cliffwalking(variation=None, invented=True):
    env = CliffWalking()
    if variation:
        env = env.vary(variation)
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 1)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 1)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp2_invent,
                                                    invented: temp_2extential,
                                                    invented3: temp2_invent,
                                                    invented4: temp2_invent,
                                                    UP: temp2_main,
                                                    DOWN: temp2_main,
                                                    LEFT: temp2_main,
                                                    RIGHT: temp2_main},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp1, DOWN: temp1, LEFT: temp1, RIGHT: temp1}, 1)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_windycliffwalking(variation=None, invented=True):
    env = WindyCliffWalking()
    if variation:
        env = env.vary(variation)
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 1)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 1)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp2_invent,
                                                    invented: temp_2extential,
                                                    invented3: temp2_invent,
                                                    invented4: temp2_invent,
                                                    UP: temp2_main,
                                                    DOWN: temp2_main,
                                                    LEFT: temp2_main,
                                                    RIGHT: temp2_main},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp1, DOWN: temp1, LEFT: temp1, RIGHT: temp1}, 1)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_unstack(variation=None, templete="reduced", all_block=False):
    env = Unstack(all_block=all_block)
    if variation:
        env = env.vary(variation)
    if templete=="full":
        maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
        inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    if templete=="reduced":
        maintemp = [RuleTemplate(1, True)]
        inventedtemp = [RuleTemplate(1, True)]
        inventedtemp2 = [RuleTemplate(1, True)]
        inventedtemp_2extential = [RuleTemplate(2, False)]
    invented = Predicate("invented", 2)
    invented4 = Predicate("invented4", 2)
    invented2 = Predicate("invented2", 1)
    invented3 = Predicate("invented3", 1)

    program_temp = ProgramTemplate([invented, invented3, invented2, invented4],
                                                          {
                                                           invented3: inventedtemp2,
                                                           invented: inventedtemp_2extential,
                                                           invented4: inventedtemp2,
                                                           invented2: inventedtemp_2extential,
                                                           MOVE: maintemp,
                                                           }, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_stack(variation=None, templete="reduced", all_block=False):
    env = Stack(initial_state=INI_STATE2, all_block=all_block)
    if variation:
        env = env.vary(variation)
    if templete=="full":
        maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
        inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    if templete=="reduced":
        maintemp = [RuleTemplate(1, True)]
        inventedtemp = [RuleTemplate(1, True)]
        inventedtemp2 = [RuleTemplate(1, True)]
        inventedtemp_2extential = [RuleTemplate(2, False)]
    invented = Predicate("invented", 2)
    invented4 = Predicate("invented4", 2)
    invented2 = Predicate("invented2", 1)
    invented3 = Predicate("invented3", 1)

    program_temp = ProgramTemplate([invented, invented3, invented2, invented4],
                                                          {
                                                           invented3: inventedtemp2,
                                                           invented: inventedtemp2,
                                                           invented4: inventedtemp2,
                                                           invented2: inventedtemp_2extential,
                                                           MOVE: maintemp,
                                                           }, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_on(variation=None, templete="reduced", all_block=False):
    env = On(all_block=all_block)
    if variation:
        env = env.vary(variation)
    if templete=="full":
        maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
        inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    if templete=="reduced":
        maintemp = [RuleTemplate(1, True), RuleTemplate(0, True)]
        inventedtemp = [RuleTemplate(1, True)]
        inventedtemp2 = [RuleTemplate(1, True)]
        inventedtemp_2extential = [RuleTemplate(2, False)]
    invented = Predicate("invented", 2)
    invented4 = Predicate("invented4", 2)
    invented2 = Predicate("invented2", 1)
    invented3 = Predicate("invented3", 1)
    invented5 = Predicate("invented5", 2)
    invented6 = Predicate("invented6", 1)

    program_temp = ProgramTemplate([invented, invented3, invented2, invented4],
                                                          {
                                                           invented3: inventedtemp2,
                                                           invented: inventedtemp2,
                                                           invented4: inventedtemp2,
                                                           invented2: inventedtemp_2extential,
                                                           MOVE: maintemp,
                                                           }, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_tictacteo(variation=None):
    env = TicTacTeo()
    if variation:
        env = env.vary(variation)
    maintemp = [RuleTemplate(1, True)]
    inventedtemp2 = [RuleTemplate(1, True)]
    inventedtemp_2extential = [RuleTemplate(2, False)]
    invented = Predicate("invented", 2)
    invented2 = Predicate("invented2", 2)
    invented3 = Predicate("invented3", 1)
    invented4 = Predicate("invented4", 1)
    program_temp = ProgramTemplate([invented, invented2, invented3, invented4],
                                   {invented:inventedtemp2,
                                    PLACE:maintemp,
                                    invented2: inventedtemp2,
                                    invented3:inventedtemp2,
                                    invented4: inventedtemp_2extential
                                    }, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

