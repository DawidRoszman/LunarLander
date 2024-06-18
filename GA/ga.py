from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gymnasium as gym
from deap import base, creator, tools, algorithms
import pickle

env = gym.make("LunarLanderContinuous-v2")
env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
print(in_dimen, out_dimen)


def model_build(in_dimen=in_dimen, out_dimen=out_dimen):
    model = Sequential()
    model.add(Dense(32, input_dim=in_dimen, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="sigmoid"))
    model.add(Dense(out_dimen, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def model_weights_as_vector(model):
    weights_vector = []

    for layer in model.layers:
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights in layer_weights:
                vector = np.reshape(l_weights, newshape=(l_weights.size))
                weights_vector.extend(vector)

    return np.array(weights_vector)


def model_weights_as_matrix(model, weights_vector):
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size

                layer_weights_vector = weights_vector[
                    start : start + layer_weights_size
                ]
                layer_weights_matrix = np.reshape(
                    layer_weights_vector, newshape=(layer_weights_shape)
                )
                weights_matrix.append(layer_weights_matrix)

                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix


def evaluate(individual, award=0):
    env.reset()
    obs1 = env.reset()
    obs1 = obs1[0]
    model = model_build()
    model.set_weights(model_weights_as_matrix(model, individual))
    done = False
    step = 0
    while (done == False) and (step <= 1000):
        obs2 = np.expand_dims(obs1, axis=0)
        selected_move1 = model.predict(obs2)
        print(selected_move1)
        obs2, reward, done, info, _ = env.step(selected_move1[0])
        print(obs2)
        award += float(reward)
        step = step + 1
        obs1 = obs2
    return (award,)


model = model_build()
ind_size = model.count_params()
print(ind_size)
print(model.summary())

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("weight_bin", np.random.uniform, -1, 1)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=ind_size
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Mean", np.mean)
stats.register("Max", np.max)
stats.register("Min", np.min)


pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)


pop, log = algorithms.eaSimple(
    pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=30, halloffame=hof, stats=stats
)


with open("lunarlander_model.pkl", "wb") as cp_file:
    pickle.dump(hof.items[0], cp_file)
