import gymnasium as gym

env = gym.make('LunarLanderContinuous-v2')
env = env.env
env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]   #Total no. of possible actions. In this case it can take 2 continous values ranging between -1 to +1


def model_build(in_dimen=in_dimen,out_dimen=out_dimen):
    model = Sequential()
    model.add(Dense(32, input_dim=in_dimen, activation='relu'))   
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(out_dimen))

    #Just like before, compilation is not required for the algorithm to run
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate(individual,award=0):
    env.reset()
    obs1 = env.reset()
    model = model_build()
    model.set_weights(model_weights_as_matrix(model, individual))
    done = False
    step = 0
    while (done == False) and (step<=1000):
        obs2 = np.expand_dims(obs1, axis=0)
        obs3 = []
        for i in range(in_dimen):  
            obs3.append(obs2[0][i])
        obs4 = np.array(obs3).reshape(-1)
        obs = np.expand_dims(obs4, axis=0)
        selected_move1 = model.predict(obs)
        obs2, reward, done, info = env.step(selected_move1[0])
        award += reward
        step = step+1
        obs1 = obs2
    return (award,)


model = model_build()
ind_size = model.count_params()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("weight_bin", np.random.uniform,-1,1)   #Initiate weights from uniform distribution between -1 to +1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)



stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Mean", np.mean)
stats.register("Max", np.max)
stats.register("Min", np.min)



pop = toolbox.population(n=100)  #n = No. of individual in a population
hof = tools.HallOfFame(1)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=30, halloffame=hof, stats=stats)
with open("lunarlander_model.pkl", "wb") as cp_file:
    pickle.dump(hof.items[0], cp_file)
