from __future__ import print_function, division, absolute_import
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import copy
from numba import jit
import pandas as pd
from core.clause import is_variable, Clause, Atom, Predicate

from collections import namedtuple
EMBEDDING_LENGTH = 5 # embedding vector length

ProofState = namedtuple("ProofState", "substitution score")
"""
substitution is list of dictionaries.
score is a vector (Tensor) representing the sucessness scores of the proof.
"""
FAIL = ProofState(None, 0)

def substitute(atom, substitutions):
    """
    substitute variables in an atom given the list of substitution pairs
    :param atom:
    :param substitution: list of binary tuples
    :return:
    """
    results = []
    for substitution in substitutions:
        results.append(atom.replace_terms(substitution))
    return results

class NeuralProver(object):
    def __init__(self, clauses, embeddings):
        """
        :param clauses: all clauses, including facts! facts are represented as a
        clause with empty body.
        """
        self._embeddings = embeddings
        self.all_symbols = embeddings.symbols
        self.symbol_dict = {symbol:i for i,symbol in enumerate(self.all_symbols)}
        self.__clauses = self.group_clauses(clauses)
        self.__var_manager = VariableManager()
        self.similarities = None
        self.para_clauses = []
        for clause in clauses:
            if clause.predicates & self._embeddings.para_predicates:
                self.para_clauses.append(clause)

    def group_clauses(self, clauses):
        """
        :param clauses:
        :return: key-group dictionaries
        """
        return {k: list(v) for k, v in itertools.groupby(clauses,
                key=(lambda c: (c.head.arity, c.head.variable_positions)))}

    @property
    def similarity_table(self):
        df = pd.DataFrame(self.similarities.numpy(), index=self.all_symbols,
                          columns=self.all_symbols)
        return df

    @staticmethod
    def get_similarities(A,B):
        """
        :param A: embedding matrix of first N symbols
        :param B: embedding matrix of second M symbols
        :return: similarity matrix with shape (N, M)
        """
        similarities = tf.exp(-tf.sqrt(
            tf.matmul(tf.reduce_sum(A ** 2, axis=1, keep_dims=True),
                      tf.ones([1, B.shape[0]]))
            + tf.matmul(tf.ones([A.shape[0], 1]),
                        tf.reduce_sum(B ** 2, axis=1, keep_dims=True),transpose_b=True)
            - 2 * tf.matmul(A, B, transpose_b=True)+1e-5))
        return similarities

    def get_rules(self):
        df = self.similarity_table
        rules = []
        for clause in self.para_clauses:
            replace_dict = {}
            confidence = 1.0
            for predicate in clause.predicates:
                if predicate not in self._embeddings.para_predicates:
                    continue
                else:
                    top2 = df.loc[predicate].sort_values(ascending=False).head(2)
                    # omit the similarity of itself
                    score = top2[1]
                    if top2.index[1].arity != predicate.arity:
                        score = 0.0
                    else:
                        replace_dict[predicate] = top2.index[1]
                    confidence *= score
            rules.append((clause.replace_predicates(replace_dict), confidence))
        return rules

    def mask_non_predicates(self, similarities, symbols):
        mask = np.zeros_like(similarities)
        for i,symbol in enumerate(symbols):
            if isinstance(symbol, str):
                mask[i,:] = 1.0
                mask[:,i] = 1.0
        return mask*np.eye(len(symbols), len(symbols))+(1-mask)*similarities

    @staticmethod
    def from_ILP(ilp, para_clauses):
        """
        construct a NTP from a ILP definition
        :return:
        """
        background = [Clause(atom,[]) for atom in ilp.background]
        embeddings = Embeddings.from_clauses(background, para_clauses)
        return NeuralProver(background+para_clauses, embeddings)

    def update_similarity(self):
        all_embeddings = self.symbols2embeddinds(self.all_symbols)
        self.similarities = self.mask_non_predicates(self.get_similarities(all_embeddings, all_embeddings),
                                                     self.all_symbols)

    def prove(self, goals, depth):
        if isinstance(goals, Atom):
            goals = [goals]
        batch_size = len(goals)
        self.update_similarity()
        initial_state = ProofState([{} for _ in range(batch_size)], tf.ones(batch_size))
        states = self.apply_rules(goals, depth, initial_state)
        scores = tf.stack([state.score for state in states if state != FAIL])
        return tf.reduce_max(scores, axis=0)

    def batch_unify(self, heads, atoms, state):
        """
        :param heads: clause heads with the same structure
        :param atoms: atoms, which all has the same structures and arities
        :param state: 
        :return: result proof states with substituted variables and new scores(ordered)
        """
        results = [None for _ in range(len(heads))]
        substitutions = [[copy.copy(dictionary) for dictionary in state.substitution] for _ in heads]
        batch_size = len(atoms)
        constants1 = [[] for _ in range(atoms[0].arity+1)]
        constants2 = [[] for _ in range(atoms[0].arity+1)]
        if heads[0].arity != atoms[0].arity:
            return [FAIL for _ in heads]

        for i,head in enumerate(heads):
            for j in range(batch_size):
                atom = atoms[j]
                for k in range(atom.arity+1):
                    if k==0:
                        symbol1 = head.predicate
                        symbol2 = atom.predicate
                    else:
                        symbol1 = head.terms[k - 1]
                        symbol2 = atom.terms[k - 1]
                    if is_variable(symbol1) and is_variable(symbol2):
                        pass
                    elif is_variable(symbol1):
                        substitutions[i][j][symbol1] = symbol2
                    elif is_variable(symbol2):
                        substitutions[i][j][symbol2] = symbol1
                    else:
                        if j==0:
                            constants1[k].append(symbol1)
                        if i==0:
                            constants2[k].append(symbol2)
        for i in reversed(range(len(constants1))):
            if not constants1[i] or not constants2[i]:
                del constants1[i]
                del constants2[i]
        position_n = len(constants1)
        scores = tf.ones([len(heads),1])*state.score
        similarities = []
        for p in range(position_n):
            indexes1 = [self.symbol_dict[symbol] for symbol in constants1[p]]
            indexes2 = [self.symbol_dict[symbol] for symbol in constants2[p]]
            # tf.sqrt here cause gradient to be nan?
            similarities.append(tf.transpose(tf.gather(tf.transpose(tf.gather(self.similarities, indexes1)), indexes2)))
            if tf.reduce_max(scores) >=0.99:
                pass
        similarities = tf.stack(similarities)
        new_scroes = tf.reduce_prod(similarities,axis=0)
        scores = scores*new_scroes
        for i in range(len(heads)):
            results[i] = ProofState(substitutions[i], scores[i])
        return results

    def symbols2embeddinds(self,symbols):
        """
        :param symbols: list of symbols. [number of symbols, embedding_length]
        :return:
        """
        return tf.stack([self._embeddings[symbol] for symbol in symbols])

    def apply_rules(self, goals, depth, state):
        """
        the or module in the original article
        :param goal: 
        :param depth: 
        :param state: 
        :return: list of states
        """
        states = []
        if not isinstance(state, ProofState):
            raise ValueError()
        all_clauses = []
        for clauses_group in self.__clauses.values():
            unified_states = []
            clauses = [self.__var_manager.activate(clause) for clause in clauses_group]
            all_clauses.extend(clauses)
            unified_states.extend(self.batch_unify([clause.head for clause in clauses],goals, state))
            for s,c in zip(unified_states,clauses):
                states.extend(self.apply_rule(c.body, depth, s))
        return states

    def apply_rule(self, body, depth, state):
        """
        the original and module.
        Loop through all atoms of the body and apply apply_rules on each atom.
        :param body: the list of subgoals
        :param depth:
        :param state:
        :return:
        """
        if not isinstance(state, ProofState):
            raise ValueError()
        if tuple(state)==tuple(FAIL):
            return [FAIL]
        if depth==0:
            return [FAIL]
        if len(body)==0:
            return [state]
        states = []
        or_states = self.apply_rules(substitute(body[0], state.substitution),
                                     depth-1, state)
        for or_state in or_states:
            states.extend(self.apply_rule(body[1:],depth,or_state))
        return states

    def loss(self, positive, negative, depth):
        prediction = self.prove(np.concatenate([positive,negative],0), depth)
        label = tf.concat([tf.ones(len(positive)), tf.zeros(len(negative))],axis=0)
        return -tf.reduce_mean(label*tf.log(prediction+1e-5)+(1.0-label)*tf.log(1-prediction+1e-5))

    def grad(self, positive, negative, depth):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(positive, negative, depth)
            weight_decay = 0.0 # This will cause local minimum?
            regularization = 0
            for weights in self._embeddings.variables:
                regularization += tf.nn.l2_loss(weights)*weight_decay
            loss_value += regularization/len(self._embeddings.variables)
        return tape.gradient(loss_value, self._embeddings.variables)

    def sample_minibatch(self, positive, negative, batch_size, ratio=0.2):
        positive_n = int(ratio*batch_size)
        negative_n = batch_size - positive_n
        positive = np.random.choice(positive, positive_n)
        negative = np.random.choice(negative, negative_n)
        return positive, negative

    def train(self, positive, negative, depth, steps, batch_size=32):
        losses = []
        optimizer = tf.train.AdamOptimizer(learning_rate=0.008)
        for i in range(steps):
            p_sample, n_sample = self.sample_minibatch(positive, negative, batch_size)
            grads = self.grad(p_sample, n_sample, depth)
            clipped_grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
            optimizer.apply_gradients(zip(clipped_grads, self._embeddings.variables),
                                      global_step=tf.train.get_or_create_global_step())
            loss_avg = self.loss(positive, negative, depth)
            losses.append(float(loss_avg.numpy()))
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
        return losses

    def all_variables(self):
        return self._embeddings.variables

    def get_str2weights(self):
        str2weights = {}
        for k,v in self._embeddings.embeddings.items():
            if isinstance(k, str):
                str2weights[k] = v
            else:
                str2weights[k.name] = v
        return str2weights

    def log(self):
        rules = self.get_rules()
        for clause, confidence in rules:
            print("{} : {}".format(clause, confidence))

# let similarities as variable
class SymbolicNeuralProver(NeuralProver):
    def __init__(self, clauses, embeddings):
        """
        :param clauses: all clauses, including facts! facts are represented as a
        clause with empty body.
        """
        super(SymbolicNeuralProver, self).__init__(clauses, embeddings)
        self.para_predicates = self._embeddings.para_predicates
        self.all_predicates = self._embeddings.predicates.union(self.para_predicates)
        self.__init_variables()

    @staticmethod
    def from_ILP(ilp, para_clauses):
        """
        construct a NTP from a ILP definition
        :return:
        """
        background = [Clause(atom,[]) for atom in ilp.background]
        embeddings = Embeddings.from_clauses(background, para_clauses)
        return SymbolicNeuralProver(background+para_clauses, embeddings)

    def all_variables(self):
        return self.similarity_parameters

    def get_str2weights(self):
        str2weights = {}
        str2weights["para_predicates"] = self.similarity_parameters
        return str2weights

    def log(self):
        pass

    def __init_variables(self):
        M = len(self.para_predicates)
        N = len(self.all_predicates)
        self.similarity_parameters = tf.get_variable("predicates_parameter", shape=[M*N - (M**2-M)/2],
                                                      initializer=tf.random_normal_initializer)
        self.pair_dict = {}
        i = 0
        for p1 in self.all_predicates:
            for p2 in self.all_predicates:
                if (p2, p1) in self.pair_dict:
                    self.pair_dict[(p1, p2)] =\
                        self.pair_dict[(p2, p1)]
                elif p1 not in self.para_predicates and p2 not in self.para_predicates:
                    continue
                else:
                    self.pair_dict[(p1, p2)] = i
                    i += 1

    def update_similarity(self):
        self.similarities = tf.nn.sigmoid(self.similarity_parameters)

    def batch_unify(self, heads, atoms, state):
        """
        :param heads: clause heads with the same structure
        :param atoms: atoms, which all has the same structures and arities
        :param state:
        :return: result proof states with substituted variables and new scores(ordered)
        """
        results = [None for _ in range(len(heads))]
        substitutions = [[copy.copy(dictionary) for dictionary in state.substitution] for _ in heads]
        batch_size = len(atoms)
        constants1 = [[] for _ in range(atoms[0].arity+1)]
        constants2 = [[] for _ in range(atoms[0].arity+1)]
        if heads[0].arity != atoms[0].arity:
            return [FAIL for _ in heads]

        for i,head in enumerate(heads):
            for j in range(batch_size):
                atom = atoms[j]
                for k in range(atom.arity+1):
                    if k==0:
                        symbol1 = head.predicate
                        symbol2 = atom.predicate
                    else:
                        symbol1 = head.terms[k - 1]
                        symbol2 = atom.terms[k - 1]
                    if is_variable(symbol1) and is_variable(symbol2):
                        pass
                    elif is_variable(symbol1):
                        substitutions[i][j][symbol1] = symbol2
                    elif is_variable(symbol2):
                        substitutions[i][j][symbol2] = symbol1
                    else:
                        if j==0:
                            constants1[k].append(symbol1)
                        if i==0:
                            constants2[k].append(symbol2)
        for i in reversed(range(len(constants1))):
            if not constants1[i] or not constants2[i]:
                del constants1[i]
                del constants2[i]
        position_n = len(constants1)
        scores = tf.ones([len(heads),1])*state.score
        similarities = []
        for p in range(position_n):
            indexes1 = [self.symbol_dict[symbol] for symbol in constants1[p]]
            indexes2 = [self.symbol_dict[symbol] for symbol in constants2[p]]
            # tf.sqrt here cause gradient to be nan?
            similarities.append(self.similarity_lookup(indexes1, indexes2))
            if tf.reduce_max(scores) >=0.99:
                pass
        similarities = tf.stack(similarities)
        new_scroes = tf.reduce_min(similarities,axis=0)
        scores = tf.minimum(scores, new_scroes)
        for i in range(len(heads)):
            results[i] = ProofState(substitutions[i], scores[i])
        return results

    def similarity_lookup(self, indexes1, indexes2):
        similarity = []
        for index1 in indexes1:
            sim_list = []
            for index2 in indexes2:
                if index1 == index2:
                    sim_list.append(1.0)
                    continue
                if isinstance(self.all_symbols[index1], str):
                    sim_list.append(0.0)
                elif self.all_symbols[index1] in self.para_predicates or\
                    self.all_symbols[index2] in self.para_predicates:
                    predicate1 = self.all_symbols[index1]
                    predicate2 = self.all_symbols[index2]
                    sim_list.append(self.similarities[self.pair_dict[(predicate1, predicate2)]])
                else:
                    sim_list.append(0.0)
            similarity.append(tf.stack(sim_list))
        return similarity



class RLProver(NeuralProver):
    def __init__(self, clauses, embeddings, depth, env):
        """
        :param clauses:
        :param embeddings:
        :param actions: action predicates
        """
        super(RLProver, self).__init__(clauses,embeddings)
        self.actions = env.actions
        self.depth = depth
        self.env = env


    @staticmethod
    def from_Env(env, para_clauses, depth):
        """
        construct a NTP from a RL enviorment
        :return:
        """
        background = [Clause(atom,[]) for atom in env.background]
        embeddings = Embeddings.from_clauses(background, para_clauses, env.actions)
        return RLProver(background+para_clauses, embeddings, depth, env)

    def deduct_all_states(self):
        states = self.env.all_states
        goals = [Atom(action, state) for action in self.actions for state in states]
        action_eval = tf.reshape(self.prove(goals, self.depth), shape=(len(self.actions), len(states)))
        return {state: action_eval[:, i] for i, state in enumerate(states)}

    def action_eval2prob(self, action_eval):
        sum_action_eval = tf.reduce_sum(action_eval)
        if sum_action_eval > 1.0:
            action_prob = action_eval / sum_action_eval
        else:
            action_prob = action_eval + (1.0 - sum_action_eval) / len(self.env.actions)
        return action_prob, sum_action_eval - 1.0

    def policy(self, state):
        goals = [Atom(action, state) for action in self.actions]
        action_eval = self.prove(goals, self.depth)
        return self.action_eval2prob(action_eval)



class VariableManager():
    def __init__(self):
        self.__max_id = 0

    def activate(self, clause):
        activated_clause = clause.assign_var_id(self.__max_id)
        self.__max_id += len(clause.variables)
        return activated_clause

class Embeddings():
    def __init__(self, predicates, para_predicates, constants, dimension=5):
        self.predicates = set(predicates)
        self.constants = set(constants)
        self.para_predicates = set(para_predicates)
        self.embeddings = {}
        for predicate in predicates.union(para_predicates):
            self.embeddings[predicate] = tf.get_variable(predicate.name, shape=[dimension], dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer())
        for constant in constants:
            self.embeddings[constant] = tf.get_variable(constant, shape=[dimension], dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())

    def __getitem__(self, key):
        return self.embeddings[key]

    @property
    def symbols(self):
        return self.embeddings.keys()

    @property
    def variables(self):
        return self.embeddings.values()

    @staticmethod
    def from_clauses(clauses, para_clauses, target_predicates=set()):
        predicates = set()
        constants = set()
        para_predicates = set()
        for clause in clauses:
            predicates.update(clause.predicates)
            constants.update(clause.constants)
        for para_clause in para_clauses:
            const = para_clause.constants
            if not const.issubset(constants):
                raise ValueError("parameterized clause shouldn't include the constants that didn't appear"
                                 "in main clauses")
            para_predicates.update(para_clause.predicates)
        predicates.update(target_predicates)
        return Embeddings(predicates, para_predicates, constants)

if __name__ == "__main__":
    tf.enable_eager_execution()
    from core.clause import str2clause,str2atom
    clause_str = ["fatherOf(abe, homer)","parentOf(homer,cart)",
                  "grandFatherOf(X,Y):-fatherOf(X,Z),parentOf(Z,Y)"]
    clauses = [str2clause(s) for s in clause_str]
    para_clauses = []
    embeddings = Embeddings.from_clauses(clauses, para_clauses)
    ntp = NeuralProver(clauses, embeddings)
    score = ntp.prove([str2atom("grandFatherOf(abe,cart)")],2)
    assert float(score) > 0.98
    score2 = ntp.prove([str2atom("grandFatherOf(homer,cart)")],2)
    score3 = ntp.prove([str2atom("parentOf(homer,homer)")],2)
    states = ntp.apply_rule([str2atom("fatherOf(cart, homer)"), str2atom("parentOf(homer,homer)")],
                            2, ProofState([set()], [1]))

    clause_str = ["fatherOf(abe, homer)","parentOf(homer,cart)"]
    para_clauses = [str2clause("grandFatherOf(X,Y):-p(X,Z),q(Z,Y)")]
    clauses = [str2clause(s) for s in clause_str]
    positive = [str2atom("grandFatherOf(abe,cart)")]
    negative = [str2atom("grandFatherOf(cart,abe)"), str2atom("grandFatherOf(abe,homer)"),
                str2atom("grandFatherOf(homer,cart)"), str2atom("grandFatherOf(cart,homer)")]
    embeddings = Embeddings.from_clauses(clauses, para_clauses)
    ntp = NeuralProver(clauses+para_clauses, embeddings)
    ntp.train(positive,negative,2,1000)
    score = ntp.prove(str2atom("grandFatherOf(abe,cart)"),2)
    score2 = ntp.prove(str2atom("grandFatherOf(cart,abe)"),2)
    score3 = ntp.prove(str2atom("grandFatherOf(abe,homer)"),2)
    score4 = ntp.prove(str2atom("grandFatherOf(homer,cart)"),2)
    score5 = ntp.prove(str2atom("grandFatherOf(cart,homer)"),2)
    score6 = ntp.prove(str2atom("grandFatherOf(homer,abe)"),2)
    print(ntp.similarity_table.to_string())
    similarity = ntp.batch_unify([str2atom("p(abe,cart)")], [str2atom("fatherOf(abe,cart)")], ProofState([set()], [1]))
    similarity2 = ntp.batch_unify([str2atom("q(abe,cart)")], [str2atom("parentOf(abe,cart)")], ProofState([set()], [1]))
    similarity3 = ntp.batch_unify([str2atom("p(abe,cart)")], [str2atom("parentOf(abe,cart)")], ProofState([set()], [1]))

    assert float(score) == 1.0


