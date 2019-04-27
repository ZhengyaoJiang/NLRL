from __future__ import print_function, division, absolute_import
from core.setup import *
from core.rl import ReinforceLearner
import ray
from ray.tune import Trainable, run_experiments
from ray.tune.schedulers import PopulationBasedTraining

class LogicRLTrainable(Trainable):
    def _setup(self):
        """

        :param config: "task": task name, "name": id of this run,
        "actor_learning_rate":
        "critic_learning_rate":
        "discounting_rate":
        "regularization_rate":
        :return:
        """
        self.timestep = 0
        self.current_value = -99.9
        config = self.config
        task = config["task"]
        name = config["name"]
        actor_learning_rate = config["actor_learning_rate"]
        critic_learning_rate = config["critic_learning_rate"]
        discounting = config["discounting"]
        critic = TableCritic(discounting, learning_rate=critic_learning_rate)
        if task == "cliffwalking":
            man, env = setup_cliffwalking()
            agent = RLDILP(man, env)
            learner = ReinforceLearner(agent, env, actor_learning_rate,
                                       discounting=discounting,
                                       batched=True, steps=6000, name=name)
        elif task == "unstack":
            man, env = setup_unstack()
            agent = RLDILP(man, env, state_encoding="atoms")
            learner = ReinforceLearner(agent, env, actor_learning_rate,
                                       discounting=discounting,
                                       batched=True, steps=6000, name=name)
        elif task == "stack":
            man, env = setup_stack()
            agent = RLDILP(man, env, state_encoding="atoms")
            learner = ReinforceLearner(agent, env, actor_learning_rate, critic=critic,
                                       discounting=discounting,
                                       batched=True, steps=6000, name=name)
        elif task == "on":
            man, env = setup_on()
            agent = RLDILP(man, env, state_encoding="atoms")
            learner = ReinforceLearner(agent, env, actor_learning_rate, critic=critic,
                                       discounting=discounting,
                                       batched=True, steps=6000, name=name)
        else:
            raise ValueError()
        self.learner = learner
        self.sess = tf.Session()
        learner.setup_train(self.sess, False)

    def _train(self):
        log = self.learner.train_step(self.sess)
        return {"episode_reward_mean": log["return"]}

    def _save(self, path):
        self.learner.save(self.sess, path)
        return path+"/xx"

    def _restore(self, path):
        self.learner.save(self.sess, path.replace("/xx",""))

    def _stop(self):
        self.sess.close()

def run(task, name=None):
    ray.init()
    import random
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=100,
        hyperparam_mutations={
            # Allow for scaling-based perturbations, with a uniform backing
            # distribution for resampling.
            "actor_learning_rate": lambda: random.uniform(0.01, 1.0),
            # Allow perturbations within this set of categorical values.
            "critic_learning_rate": lambda: random.uniform(0.01, 1.0),
            "discounting": [0.8, 0.9, 0.95, 1.0],
        })

    # Try to find the best factor 1 and factor 2
    run_experiments(
        {
            "pbt_test3": {
                "run": LogicRLTrainable,
                "stop": {
                    "training_iteration": 8000
                },
                "num_samples": 6,
                "config": {
                    "task": task,
                    "name": name,
                    "actor_learning_rate": 0.1,
                    "critic_learning_rate": 0.1,
                    "discounting": 1.0
                },
                "trial_resources": {
                    "cpu": 2,
                },
            },
        },
        scheduler=pbt,
        verbose=False)