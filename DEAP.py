from deap import base, creator
import random
from deap import tools
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from statsmodels.tsa.arima_model import ARIMA

p=0
d=1
q=0
timeseries = []
def main():
    de = Deap()
    print(de)
def read_file(filename):
    dataframe = pd.read_csv(filename,usecols=[2,9,14,21],names=None,encoding='ISO-8859-1',header=None,low_memory=False)
    dataframe = dataframe[1:]
    return dataframe

def evaluate(individual):
    predicterror = ARIMA_Model(individual[0],individual[1])
    return predicterror,

def Deap():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    IND_SIZE = 2  # 种群数
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.randint,0,6)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)  # mate:交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # mutate : 变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
    toolbox.register("evaluate", evaluate)  # commit our evaluate

    pop = toolbox.population(n=10)
    print(pop)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 10
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))
    print("-- Iterative %i times --" % NGEN)

    for g in range(NGEN):
        if g % 10 == 0:
            print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        print(len(offspring))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Change map to list,The documentation on the official website is wrong

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        print(len(pop))
    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    return best_ind, best_ind.fitness.values

def estimating(timeseries):
    ts_log = np.log(timeseries)
    return ts_log

def ARIMA_Model(p,q):
    # year = input('input the year you want to look:')
    filename = ('2013.csv')

    dataframe = read_file(filename)
    sdate = dataframe[2]
    new_sdate = []
    flag = dataframe[9]
    flag = np.asarray(flag).astype(float).astype(int)
    goods = dataframe[14]
    goods = np.asarray(goods)
    num = dataframe[21]
    num = abs(np.asarray(num).astype(float))
    new_num = []
    for i in range(len(sdate)):
        if goods[i] == '2996' and flag[i] == 3:
            new_num.append(num[i])
            tmp = sdate[i].split(' ')
            if len(tmp) != 0:
                new_sdate.append(tmp[0])
            else:
                new_sdate.append(0)
                print('DATE SPLIT ERROR!')
    data = Date_Differ(new_sdate, new_num)

    f_data = DataFrame(data, columns=['date', 'num'])
    f_data['num'] = f_data['num'].convert_objects(convert_numeric=True)
    timeseries = f_data['num']
    ts_log = np.log(timeseries)
    print(type(ts_log))
    model = ARIMA(ts_log, order=(p, d, q))
    result_ARIMA = model.fit(disp=-1)
    # ts_log_diff = Diff(timeseries)

    predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    predictionsRMSE = np.sqrt(sum((predictions_ARIMA - timeseries) ** 2) / len(timeseries))
    return predictionsRMSE
    # plt.plot(ts)
    # plt.plot(predictions_ARIMA)
    # plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
    # plt.show()


def Date_Differ(date_list,num_list):
    sum = 0
    newdate = []
    newnum = []
    for i in range(len(date_list)):
        if i == 0:
            newdate.append(date_list[0])
            sum = num_list[0]
        else:
            if date_list[i] != date_list[i - 1]:
                newdate.append(date_list[i])
                newnum.append(sum)
                sum = 0
            sum = sum + num_list[i]
            if i == len(date_list) - 1:
                newnum.append(sum)
    date_num =np.asarray([])
    for i in range(len(newdate)):
        if i == 0:
            date_num=np.asarray([newdate[0],newnum[0]]).reshape((1,2))
        else:
            tmp_ = date_num
            date_num = np.vstack((tmp_,[newdate[i],newnum[i]]))
    #print(type(newnum[10]))
    return date_num


if __name__ == "__main__":
    main()
