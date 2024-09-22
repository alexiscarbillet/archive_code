# author: ALEXIS CARBILLET

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

# datasets
pma343 = pd.read_csv('pma343.csv', sep=';')
xqf131 = pd.read_csv('xqf131.csv', sep=';')
xqg237 = pd.read_csv('xqg237.csv', sep=';')


pma343.columns = ['x', 'y']
xqf131.columns = ['x', 'y']
xqg237.columns = ['x', 'y']



class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
        
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0    
    
    def routeDistance(self,route):
        for i in range(1,len(route)):
            self.distance+=distances[cityList.index(route[i])][cityList.index(route[i-1])]
        self.distance+=distances[cityList.index(route[0])][cityList.index(route[len(route)-1])]
        return self.distance
    
    def routeFitness(self,route):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance(route))
        return self.fitness
        
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    # distances=[]
    # n=len(cityList)
    # for i in range(n):
    #     distances.append([])
    #     for j in range(n):
    #         xDis = abs(cityList[i].X - cityList[j].x)
    #         yDis = abs(cityList[i].y - cityList[j].y)
    #         distance = np.sqrt((xDis ** 2) + (yDis ** 2))
    #         distances[i].append(distance)
    # print(distances)
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness(population[i])
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults
    
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
    
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child
    
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children
    
def mutate(individual, mutationRate):
    if random.random() < mutationRate:
        swapped = int(random.random() * len(individual))
        u=int(random.random() * len(individual))
        while u == swapped:
            u=int(random.random() * len(individual))
        swapWith = u
                
        city1 = individual[swapped]
        city2 = individual[swapWith]
        
        individual[swapped] = city2
        individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop
    
def nextGeneration(currentGen, eliteSize, generations,i):
    # mutationRate=1-i/generations
    mutationRate=0.2
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration
    
def geneticAlgorithm(population, popSize, eliteSize, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, generations,i)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute
    
cityList=[]
for i in range(len(pma343)):
    cityList.append(City(x=pma343['x'][i], y=pma343['y'][i]))
    
# geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, generations=500)


def geneticAlgorithmPlot(population, popSize, eliteSize, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    m=pop
    dis=progress[0]
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, generations,i)
        progress.append(1 / rankRoutes(pop)[0][1])
        if progress[-1]<dis:
            bestRouteIndex = rankRoutes(pop)[0][0]
            m = pop[bestRouteIndex]
            dis=progress[-1]
            
    plt.figure()
    plt.plot(progress)
    plt.title('Distance obtained according to the number of generation')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.figure()
    # plt.plot(pma343['x'],pma343['y'],'o')
    print('Minumum obtained at the: ',progress.index(dis),' genetration with a distance of ',dis)
    print(m) 
    x=[]
    y=[]
    for i in m:
        x.append(i.x)
        y.append(i.y)
    plt.plot(x,y,'-o')
    plt.title('Route')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.show()
    
# cityList=[]
# distances=[]
# n=len(pma343)
# for i in range(n):
#     cityList.append(City(x=pma343['x'][i], y=pma343['y'][i]))
#     distances.append([])
#     for j in range(n):
#         distances[i].append(np.sqrt(np.abs(pma343['x'][i]-pma343['x'][j])**2+np.abs(pma343['y'][i]-pma343['y'][j])))
# geneticAlgorithmPlot(population=cityList, popSize=300, eliteSize=20, mutationRate=0.01, generations=600)

cityList=[]
distances=[]
n=len(xqf131)
for i in range(n):
    cityList.append(City(x=xqf131['x'][i], y=xqf131['y'][i]))
    distances.append([])
    for j in range(n):
        distances[i].append(np.sqrt(np.abs(xqf131['x'][i]-xqf131['x'][j])**2+np.abs(xqf131['y'][i]-xqf131['y'][j])))
geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, generations=3000)
    
# cityList=[]
# distances=[]
# n=len(xqg237)
# for i in range(n):
#     cityList.append(City(x=xqg237['x'][i], y=xqg237['y'][i]))
#     distances.append([])
#     for j in range(n):
#         distances[i].append(np.sqrt(np.abs(xqg237['x'][i]-xqg237['x'][j])**2+np.abs(xqg237['y'][i]-xqg237['y'][j])))
# geneticAlgorithmPlot(population=cityList, popSize=300, eliteSize=20, generations=600) 