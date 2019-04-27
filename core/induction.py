from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import pandas as pd
import tensorflow.contrib.eager as tfe
import os
from core.rules import RulesManager
from core.clause import Predicate
from pprint import pprint

class BaseDILP(object):
    def __init__(self, rules_manager, background, independent_clause=True, scope_name="rule_weights"):
        self.rules_manager = rules_manager
        self.independent_clause = independent_clause
        self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices
        self.__init__rule_weights(scope_name)
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(background)
        self._construct_graph()
        self.previous_definition = {}

    def _construct_graph(self):
        self.tf_input_valuation = tf.placeholder(shape=[None, self.base_valuation.shape[0]], dtype=tf.float32)
        self.tf_result_valuation = self._construct_deduction()

    def __init__rule_weights(self, scope_name="rule_weights"):
        if self.independent_clause:
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                for predicate, clauses in self.rules_manager.all_clauses.items():
                    self.rule_weights[predicate] = []
                    for i in range(len(clauses)):
                        self.rule_weights[predicate].append(tf.get_variable(predicate.name+"_rule_weights"+str(i),
                                                                    [len(clauses[i]),],
                                                                    initializer=tf.random_normal_initializer,
                                                                    dtype=tf.float32))
        else:
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                for predicate, clauses in self.rules_manager.all_clauses.items():
                    self.rule_weights[predicate] = tf.get_variable(predicate.name + "_rule_weights",
                                                                   [len(clauses[0]), len(clauses[1])],
                                                                   initializer=tf.random_normal_initializer,
                                                                   dtype=tf.float32)

    def show_definition(self, sess):
        for predicate in self.rules_manager.all_clauses:
            print()
            print(predicate)
            result = self.get_predicate_definition(sess, predicate)
            for weight, clause in result:
                if weight>0.1:
                    pprint(str(round(weight, 3))+": "+clause)
            print("------------------------------------")
            print("differences:")
            if predicate in self.previous_definition:
                for i,(weight, clause) in enumerate(result):
                    difference = weight - self.previous_definition[predicate][i][0]
                    if abs(difference)>1e-3:
                        print(str(round(difference, 3))+": "+clause)
            print("=======================================")
            self.previous_definition[predicate] = result

    def get_predicates_definition(self, sess, threshold=0.0):
        result = {}
        for predicate in self.rules_manager.all_clauses:
            result[predicate] = self.get_predicate_definition(sess, predicate, threshold)
        return result


    def get_predicate_definition(self, sess, predicate, threshold=0.0):
        clauses = self.rules_manager.all_clauses[predicate]
        rules_weights = self.rule_weights[predicate]
        rules_weights = sess.run([rules_weights])[0]
        result = []
        for i, rule_weights in enumerate(rules_weights):
            weights = softmax(rule_weights)
            indexes = np.nonzero(weights>threshold)[0]
            for j in range(len(indexes)):
                result.append((weights[indexes[j]], str(clauses[i][indexes[j]])))
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

    def deduction(self, state=None, session=None):
        # takes background as input and return a valuation of target ground atoms
        if not state:
            valuation = self.base_valuation
        else:
            valuation = self.base_valuation+self.axioms2valuation(state)
        if session:
            result = session.run([self.tf_result_valuation], feed_dict={self.tf_input_valuation:[valuation]})[0]
        else:
            with tf.Session() as sess:
                result = sess.run([self.tf_result_valuation], feed_dict={self.tf_input_valuation:[valuation]})[0]
        return result[0]

    def _construct_deduction(self):
        valuation = tf.transpose(self.tf_input_valuation)
        for _ in range(self.rules_manager.program_template.forward_n):
            valuation = self.inference_step(valuation)
        return tf.transpose(valuation)

    def inference_step(self, valuation):
        deduced_valuation = tf.zeros_like(valuation)
        # deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate, matrix in self.rules_manager.deduction_matrices.items():
            deduced_valuation += BaseDILP.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
        return deduced_valuation+tf.transpose(self.tf_input_valuation)
        #return prob_sum(deduced_valuation, valuation)

    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''
        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = [[] for _ in rule_weights]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(BaseDILP.inference_single_clause(valuation, matrix))

        c_p = None
        for i in range(len(result_valuations)):
            valuations = tf.stack(result_valuations[i])
            prob_rule_weights = tf.nn.softmax(rule_weights[i])[:, None, None]
            if c_p==None:
                c_p = tf.reduce_sum(prob_rule_weights*valuations, axis=0)
            else:
                c_p = prob_sum(c_p, tf.reduce_sum(prob_rule_weights*valuations, axis=0))
        return c_p

    @staticmethod
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        X1 = X[:, :, 0, None]
        X2 = X[:, :, 1, None]
        Y1 = tf.gather_nd(params=valuation, indices=X1)
        Y2 = tf.gather_nd(params=valuation, indices=X2)
        Z = Y1*Y2
        return tf.reduce_max(Z, axis=1)

    def all_variables(self):
        if self.independent_clause:
            return [weight for weights in self.rule_weights.values() for weight in weights]
        else:
            return [weights for weights in self.rule_weights.values()]



class SupervisedDILP(BaseDILP):
    def __init__(self, rules_manager, ilp, learning_rate=0.5):
        super(SupervisedDILP, self).__init__(rules_manager, ilp.background)
        self.training_data = OrderedDict() # index to label
        self.__init_training_data(ilp.positive, ilp.negative)
        self.learning_rate=learning_rate
        self._construct_train()

    def __init_training_data(self, positive, negative):
        for i, atom in enumerate(self.ground_atoms):
            if atom in positive:
                self.training_data[i] = 1.0
            elif atom in negative:
                self.training_data[i] = 0.0

    def _construct_train(self):
        self.tf_loss = self.loss()
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        grads, loss = self.grad()
        self.tf_train = optimizer.apply_gradients(zip(grads, self.all_variables()),
                                  global_step=tf.train.get_or_create_global_step())

    def loss(self, batch_size=-1):
        labels = np.array(self.training_data.values(), dtype=np.float32)
        outputs = tf.gather(self.tf_result_valuation[0], np.array(self.training_data.keys(), dtype=np.int32))
        if batch_size>0:
            index = tf.random_uniform([batch_size], 0, len(labels))
            labels = labels[index]
            outputs = tf.gather(outputs, index)
        loss = -tf.reduce_mean(labels*tf.log(outputs+1e-10)+(1-labels)*tf.log(1-outputs+1e-10))
        return loss

    def grad(self):
        loss_value = self.loss(-1)
        weight_decay = 0.0
        regularization = 0
        for weights in self.all_variables():
            weights = tf.nn.softmax(weights)
            regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
        loss_value += regularization/len(self.all_variables())
        return tf.gradients(loss_value, self.all_variables()), loss_value

    def train(self, steps=300, name=None):
        """
        :param steps:
        :param name:
        :return: the loss history
        """
        if self.independent_clause:
            str2weights = {str(key) + str(i): value[i] for key, value in self.rule_weights.items() for i in
                           range(len(value))}
        else:
            str2weights = {str(key): value for key, value in self.rule_weights.items()}

        if name:
            checkpoint = tf.train.Checkpoint(**str2weights)
            checkpoint_dir = "./model/"+name
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            try:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            except Exception as e:
                print(e)

        losses = []

        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables()])
            for i in range(steps):
                _, loss = sess.run([self.tf_train, self.tf_loss], feed_dict={self.tf_input_valuation:[self.base_valuation]})
                loss_avg = float(loss)
                losses.append(loss_avg)
                print("-"*20)
                print("step "+str(i)+" loss is "+str(loss_avg))
                if i%10==0:
                    self.show_definition(sess)
                    valuation_dict = self.valuation2atoms(self.deduction(session=sess)).items()
                    for atom, value in valuation_dict:
                        print(str(atom)+": "+str(value))
                    if name:
                        checkpoint.save(checkpoint_prefix)
                        pd.Series(np.array(losses)).to_csv(name+".csv")
                print("-"*20+"\n")
        return losses

class RLDILP(BaseDILP):
    def __init__(self, rules_manager, env, independent_clause=True, state_encoding="atoms"):
        super(RLDILP, self).__init__(rules_manager, env.background, independent_clause)
        self.env = env
        self.state_encoding=state_encoding
        if self.state_encoding=="atoms":
            self.all_actions = self.get_all_actions()
        else:
            self.all_actions = self.env.actions

    def get_all_actions(self):
        atoms = self.valuation2atoms(self.base_valuation, -1).keys() #ordered
        actions = []
        for atom in atoms:
            if atom.predicate in self.env.actions:
                actions.append(atom)
        return actions

    def get_valuation_indexes(self, state=None):
        """
        :param state: tuple of terms, if encoding is atoms it is not need to be feed
        :return: the indexes of valuations of action atoms
        """
        atoms = self.valuation2atoms(self.base_valuation, -1).keys() #ordered
        indexes = [None for _ in range(len(self.all_actions))]
        for i,atom in enumerate(atoms):
            if self.state_encoding == "terms":
                if state[0].terms == atom.terms and atom.predicate in self.env.actions:
                    indexes[self.all_actions.index(atom.predicate)] = i
            else:
                if atom in self.all_actions:
                    indexes[self.all_actions.index(atom)] = i
        return np.array(indexes)

    def get_str2weights(self):
        if self.independent_clause:
            str2weights = {str(key) + str(i): value[i] for key, value in self.rule_weights.items() for i in
                           range(len(value))}
        else:
            str2weights = {str(key): value for key, value in self.rule_weights.items()}
        return str2weights

    def create_checkpoint(self, name):
        if name:
            checkpoint = tfe.Checkpoint(**self.get_str2weights())
            checkpoint_dir = "./model/"+name
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            try:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            except Exception as e:
                print(e)
            return checkpoint, checkpoint_prefix
        else:
            return None, None


    def log(self, sess):
        self.show_definition(sess)
        valuation_dict = self.valuation2atoms(self.deduction(session=sess)).items()
        for atom, value in valuation_dict:
            print(str(atom)+": "+str(value))



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x =  x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def prob_sum(x, y):
    return x + y - x*y
