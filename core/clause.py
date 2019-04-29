from __future__ import print_function, division, absolute_import
import numpy as np
from collections import namedtuple

Predicate = namedtuple("Predicate", "name arity")

def is_variable(term):
    return isinstance(term, int)

def var_string(atom):
    '''
    find all variable string (string with first letter capitalized)
    :param atom: atom where variables not replaced with integers
    :return: set of variables string
    '''
    vars = set()
    for term in atom.terms:
        if term[0].isupper():
            vars.add(term)
    return vars

def str2atom(s):
    '''
    :param s: 
    :return: Atom where variables not replaced with integers
    '''
    s = s.replace(" ", "")
    left = s.find("(")
    right = s.find(")")
    terms = s[left + 1:right].split(",")
    predicate = Predicate(s[:left], len(terms))
    return Atom(predicate, terms)

def str2clause(s):
    """
    :param s: 
    :return: 
    """
    s = s.replace(" ", "").replace(".", "")
    atoms = s.split(":-")
    head_str = atoms[0]
    head = str2atom(head_str)
    if len(atoms) ==2:
        body_strs = atoms[1].replace("),", ") ").split(" ")
        body = [str2atom(s) for s in body_strs]
        clause = Clause(head, body)
        var_strs = set()
        for atom in body+[head]:
            print(atom)
        for strs in [var_string(atom) for atom in body+[head]]:
            var_strs = var_strs.union(strs)
        return clause.replace_by_dict({s: i for i,s in enumerate(var_strs)})
    else:
        return Clause(head, [])


class Atom(object):
    def __init__(self, predicate, terms):
        '''
        :param predicate: Predicate, the predicate of the atom
        :param terms: tuple of string (or integer) of size 1 or 2.
        use integer 0, 1, 2 as variables
        '''
        object.__init__(self)
        self.predicate = predicate
        self.terms = tuple(terms)
        assert len(terms) == predicate.arity

    @property
    def arity(self):
        return len(self.terms)

    def __hash__(self):
        hashed_list = list(self.terms[:])
        hashed_list.append(self.predicate)
        return hash(tuple(hashed_list))

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        terms_str = ""
        variable_table = ["X", "Y", "Z", "M", "N"]
        for term in self.terms:
            if isinstance(term, int):
                terms_str += variable_table[term]
            else:
                terms_str += term
            terms_str += ","
        terms_str = terms_str[:-1]
        return self.predicate.name+"("+terms_str+")"

    @property
    def variables(self):
        var = [symbol for symbol in self.terms if isinstance(symbol, int)]
        return set(var)

    @property
    def variable_positions(self):
        pos = [i for i,symbol in enumerate(self.terms) if isinstance(symbol, int)]
        return tuple(pos)

    @property
    def constants(self):
        const = [symbol for symbol in self.terms if isinstance(symbol, str)]
        return set(const)

    def match_variable(self, target):
        '''
        :param target: ground atom to be matched
        :return: dictionary from int to string, indicating the map from variable to constant. return empty dictionary if
        the two cannot match.
        '''
        assert self.predicate == target.predicate, str(self.predicate)+" and "+str(target.predicate)+" can not match"
        match = {}
        for i in range(self.arity):
            if isinstance(self.terms[i], str):
                if self.terms[i] == target.terms[i]:
                    continue
                else:
                    return {}
            else:
                match[self.terms[i]] = target.terms[i]
        return match

    def replace_terms(self, match):
        '''
        :param match: match dictionary
        :return: a atoms whose variable is replaced by constants, given the match mapping.
        '''
        terms = []
        for i,variable in enumerate(self.terms):
            if variable not in match:
                terms.append(variable)
            else:
                terms.append(match[variable])
        result = Atom(self.predicate, terms)
        return result

    def replace_predicate(self, predicate_dict):
        for k,v in predicate_dict.items():
            if k == self.predicate:
                return Atom(v, self.terms)
        return self

    def normalized_atom(self, id):
        symbols = []
        for symbol in self.terms:
            if isinstance(symbol, int):
                symbols.append(symbol-id)
            else:
                symbols.append(symbol)
        return Atom(self.predicate, symbols)

    def assign_var_id(self, start):
        var_map = {}
        for symbol in self.terms:
            if isinstance(symbol, int):
                var_map[symbol] = start+symbol
        return self.replace_terms(var_map)


class Clause():
    def __init__(self, head, body):
        '''
        :param head: atom, result of a clause
        :param body: list of atoms, conditions, amximum length is 2.
        '''
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ""
        min_varible = min(self.variables)
        new_head = self.head.normalized_atom(min_varible)
        new_body = [body_atom.normalized_atom(min_varible) for body_atom in self.body]
        for body_atom in new_body:
            body_str += str(body_atom)
            body_str += ","
        body_str = body_str[:-1]
        return str(new_head)+":-"+body_str

    def replace_by_head(self, head):
        '''
        :param head: a ground atom
        :return: replaced clause
        '''
        match = self.head.match_variable(head)
        new_body = []
        for atom in self.body:
            new_body.append(atom.replace_terms(match))
        return Clause(head, new_body)

    def replace_predicates(self, predicates_dict):
        new_head = self.head.replace_predicate(predicates_dict)
        new_body = [atom.replace_predicate(predicates_dict) for atom in self.body]
        return Clause(new_head, new_body)


    def replace_by_dict(self, match):
        head = self.head.replace_terms(match)
        body = [atom.replace_terms(match) for atom in self.body]
        return Clause(head, body)

    @property
    def atoms(self):
        return [self.head]+list(self.body)

    @property
    def variables(self):
        return set().union(*[atom.variables for atom in self.atoms])

    @property
    def constants(self):
        return set().union(*[atom.constants for atom in self.atoms])

    @property
    def predicates(self):
        return set([atom.predicate for atom in self.atoms])

    def __hash__(self):
        hashed_list = list(self.body.copy())
        hashed_list.append(self.head)
        return hash(tuple(hashed_list))

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def assign_var_id(self, start):
        head = self.head.assign_var_id(start)
        body = [atom.assign_var_id(start) for atom in self.body]
        return Clause(head, body)

