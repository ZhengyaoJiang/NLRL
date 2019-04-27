from __future__ import print_function, division, absolute_import
import numpy as np

class ILP():
    '''
    a Inductive logic programming problem
    predicates are defined as strings, so as the constants
    '''
    def __init__(self, language_frame,background, positive, negative):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.language_frame = language_frame
        self.background = background
        self.positive = positive
        self.negative = negative

class LanguageFrame():
    def __init__(self, target, extensional, constants):
        '''
        :param target: target predicate, list of target predicates
        :param extensional: list of Predicates, extensional predicates and their arity
        :param constants: list of strings, constants
        '''
        if not isinstance(target, list):
            target = [target]
        self.target = target
        self.extensional = extensional
        self.constants = constants

class RuleTemplate():
    def __init__(self, variables_n, allow_intensional):
        '''
        :param variables_n: integer, number of
        existentially quantified variables allowed in the clause
        :param allow_intensional: boolean, whether the atoms in the body
         of the clause can use intensional predicates
        '''
        self.variables_n = variables_n
        self.allow_intensional = allow_intensional

class ProgramTemplate():
    def __init__(self, auxiliary, rule_temps, forward_n):
        '''
        :param auxiliary: list of Predicates, set of auxiliary intensional predicates and their arity
        :param rule_temps: dictionary of predicate to tuples of rule templates,
        map from each intensional predicate to a pair of rule templates
        :param forward_n: integer4, max number of steps of forward chaining
        '''
        self.auxiliary = auxiliary
        self.rule_temps = rule_temps
        self.forward_n = forward_n


