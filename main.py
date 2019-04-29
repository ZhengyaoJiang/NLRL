from core.setup import *
from collections import OrderedDict
import argparse
import json

def generalized_test(task, name, algo):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if task == "cliffwalking":
        env = CliffWalking
    elif task == "unstack":
        env = Unstack
    elif task == "stack":
        env = Stack
    elif task == "on":
        env = On
    elif task == "windycliffwalking":
        env = WindyCliffWalking
    else:
        raise ValueError()
    import tensorflow as tf
    summary = OrderedDict()
    if algo=="DILP":
        starter = start_DILP
    elif algo=="NN":
        starter = start_NN
    elif algo=="Random":
        starter = start_Random
    variations = env.all_variations

    for variation in [""]+list(variations):
        tf.reset_default_graph()
        print("==========="+variation+"==============")
        result = starter(task, name, "evaluate", variation)
        pprint(result)
        variation = "train" if not variation else variation
        summary[variation] = {"mean":round(result["mean"], 3), "std": round(result["std"], 3),
                              "distribution":result["distribution"]}
    for k,v in summary.items():
        print(k+": "+str(v["mean"])+"+-"+str(v["std"]))
    with open("model/"+name+"/result.json", "wr") as f:
        json.dump(summary, f)

def start_Random(task, name, mode, variation=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if task == "cliffwalking":
        man, env = setup_cliffwalking(variation)
        agent = RandomAgent(env.action_n)
        discounting = 1.0
        critic = None
        learner = ReinforceLearner(agent, env, 0.1, critic=critic, discounting=discounting,
                                   batched=True, steps=12000, name=name)
    if task == "windycliffwalking":
        man, env = setup_windycliffwalking(variation)
        agent = RandomAgent(env.action_n)
        discounting = 1.0
        critic = None
        learner = ReinforceLearner(agent, env, 0.1, critic=critic, discounting=discounting,
                                   batched=True, steps=12000, name=name)
    elif task == "unstack":
        man, env = setup_unstack(variation)
        agent = RandomAgent(env.action_n)
        critic = None
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=50000, name=name)
    elif task == "stack":
        man, env = setup_stack(variation)
        agent = RandomAgent(env.action_n)
        critic = None
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=50000, name=name)
    elif task == "on":
        man, env = setup_on(variation)
        agent = RandomAgent(env.action_n)
        critic = None
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=50000, name=name)
    else:
        raise ValueError()
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()


#@ray.remote
def start_DILP(task, name, mode, variation=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if task == "cliffwalking":
        man, env = setup_cliffwalking(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        discounting = 1.0
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector,
                                  involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic, discounting=discounting,
                                   batched=True, steps=50000, name=name)
    if task == "windycliffwalking":
        man, env = setup_windycliffwalking(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        discounting = 1.0
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector,
                                  involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic, discounting=discounting,
                                   batched=True, steps=50000, name=name)
    elif task == "unstack":
        man, env = setup_unstack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=30000, name=name)
    elif task == "stack":
        man, env = setup_stack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=30000, name=name)
    elif task == "on":
        man, env = setup_on(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=30000, name=name)
    else:
        raise ValueError()
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()

def start_NN(task, name, mode, variation=None):
    if task == "cliffwalking":
        man, env = setup_cliffwalking(variation)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.002, critic=critic,
                                   steps=120000, name=name)
    if task == "windycliffwalking":
        man, env = setup_windycliffwalking(variation)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.002, critic=critic,
                                   steps=120000, name=name)
    elif task == "stack":
        man, env = setup_stack(variation, all_block=True)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        if variation:
            critic = None
        else:
            # critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
            # critic = None
            #critic = TableCritic(discounting=1.0, learning_rate=0.01, involve_steps=True)
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.002, critic=critic,
                                   steps=120000, name=name)
    elif task == "unstack":
        man, env = setup_unstack(variation, all_block=True)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        #critic = None
        # critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
        if variation:
            critic = None
        else:
            # critic = None
            #critic = TableCritic(discounting=1.0, learning_rate=0.01, involve_steps=True)
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.002, critic=critic,
                                   steps=120000, name=name)
        #learner = PPOLearner(agent, env, 0.5, critic=critic, steps=50000, name=name)
    elif task == "on":
        man, env = setup_on(variation, all_block=True)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        # critic = TableCritic(1.0)
        if variation:
            critic = None
        else:
            # critic = None
            #critic = TableCritic(discounting=1.0, learning_rate=0.01, involve_steps=True)
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001, state2vector=env.state2vector)
            # critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.002, critic=critic,
                                   steps=120000, name=name)
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()

from pprint import pprint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--task')
    parser.add_argument('--algo')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if args.mode=="generalize":
        generalized_test(args.task, args.name, args.algo)
    elif args.mode=="train":
        try:
            if args.algo == "DILP":
                starter = start_DILP
            elif args.algo == "NN":
                starter = start_NN
            else:
                raise ValueError()
            pprint(starter(args.task, args.name, args.mode))
        except Exception as e:
            print(e.message)
            raise e
        finally:
            generalized_test(args.task, args.name, args.algo)

