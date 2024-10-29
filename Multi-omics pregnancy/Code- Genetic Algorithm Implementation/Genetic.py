#!/usr/bin/env python
# coding: utf-8

# In[]:
!pip install xlrd==1.2.0


# In[13]:


import time
import xlrd
import xlsxwriter
import csv
import random
import numpy as np
from random import randint
from numpy import mean
from numpy import absolute
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score


# In[14]:


row = 0
column = 0


# In[15]:


size_pop = 10
size_gen = 10
cross_rate = 0.8
mutation_rate = 0.2
size_cross = cross_rate*size_pop
N = int(size_pop-size_cross)


# In[16]:


def read_data():
    with open('ImmuneSystem.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        label = []
        data = []
        row = 0
        line_num = 0
        for row_data in csv_reader:

            if line_num == 0:
                column = len(row_data) - 1
                size_pop = column//5

            else:
                label.append(row_data[0])
                data.append(row_data[1:])

            line_num += 1

    for i in range(len(label)):
        label[i] = label[i].replace('BL', '4')
        t = label[i].split('_')
        label[i] = t[1]

    data = np.array(data)
    label = np.array(label)

    return data, label


# In[17]:


def individual(lenght):
    chromosome = []
    for i in range(lenght):
        chromosome.append(np.random.choice([0, 1], p=[0.8, 0.2]))

    #chromosome=[random.randint(0,1) for i in range(lenght)]
    # print(chromosome)
    return chromosome


# In[18]:


def fit_data(bv, samples):
    fit_samples = []

    for i in range(len(bv)):
        if bv[i] == 1:
            fit_samples.append(samples[:, i])

    return fit_samples


# In[19]:


def compute_fitness(fit_samples, label):

    fit_samples = np.array(fit_samples)
    fit_samples = np.transpose(fit_samples)

    #X_train, X_test, y_train, y_test = train_test_split(fit_samples, label, test_size=0.33)
    model = ElasticNet(alpha=1.0, l1_ratio=0.01)
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=1)
    scores = cross_val_score(model, fit_samples, label,
                             scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = absolute(scores)
    f = mean(scores)

    # enet.fit(X_train,y_train)
    # y_pred=enet.predict(X_test)
    #f=accuracy_score(y_test, y_pred, normalize=False)

    return f


# In[20]:


def first_population(samples, label):
    population_list = []

    for i in range(size_pop):
        binary_vector = individual(len(samples[0]))
        # print(binary_vector)

        # print(samples)

        fit_samples = fit_data(binary_vector, samples)

        f = compute_fitness(fit_samples, label)
        # print(f)
        # print(f"{f}/{binary_vector}")
        population_list.append((binary_vector, f))

    return population_list


# In[21]:


def cross(population, samples, label):
    new_population = []
    for _ in range(int(size_cross//2)):
        v1 = randint(0, len(population)-1)
        v2 = randint(0, len(population)-1)

        mid = randint(1, len(population[v1][0])-2)
        child1 = population[v1][0][:mid]+population[v2][0][mid:]
        child2 = population[v2][0][:mid]+population[v1][0][mid:]

        # print(f'v1={population[v1][0]}\nv2={population[v2][0]}\nmid={mid}')
        # print(f'child1={child1}\nlen={len(population[v1][0])==len(child1)}')
        #print(f'check sides\nv1={population[v1][0][:mid]==child1[:mid]}\nv2={population[v2][0][mid:]==child1[mid:]}')

        # print(f'child2={child2}\nlen={len(population[v1][0])==len(child2)}')
        #print(f'check sides\nv1={population[v2][0][:mid]==child2[:mid]}\nv2={population[v1][0][mid:]==child2[mid:]}')

        fit_samples = fit_data(child1, samples)

        fitness = compute_fitness(fit_samples, label)

        new_population.append((child1, fitness))

        # child2

        fit_samples = fit_data(child2, samples)

        fitness = compute_fitness(fit_samples, label)

        new_population.append((child2, fitness))

    #new_population.sort(key=lambda tup:tup[1])

    return new_population


# In[22]:


def mutant(population, samples, labels, rate=0.5):
    new_population = []
    for p in population:
        mut = random.uniform(0, 1)
        if mut < mutation_rate:
            chromosom = []
            for ch in p[0]:
                if random.uniform(0, 1) > rate:
                    if ch == 1:
                        chromosom.append(0)
                    else:
                        chromosom.append(1)
                else:
                    chromosom.append(ch)

            fit_samples = fit_data(chromosom, samples)
            fitness = compute_fitness(fit_samples, labels)
            if fitness < p[1]:
                new_population.append((chromosom, fitness))
            else:
                new_population.append(p)
        else:
            new_population.append(p)
    return new_population


# In[23]:


def read_feature():
    features = []
    with open("ImmuneSystem.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        features = next(csv_reader)
        features.pop(0)
        # print(features[0])
    return features


# In[8]:

import xlrd
def check_feature(individual_feature):
    selected_features = []
    wb = xlrd.open_workbook("ImmuneSystem.xlsx")
    sheet = wb.sheet_by_index(0)
    for i in range(sheet.nrows):
        selected_features.append(sheet.cell_value(i, 0))

    detected_features = list(filter(lambda x: x in selected_features, individual_feature))
    not_detected_features = list(filter(lambda x: x not in individual_feature, selected_features))
    extera_features = list(filter(lambda x: x not in selected_features, individual_feature))

    return {"detected": detected_features, "not_detected": not_detected_features, "extera": extera_features}

# In[24]:
import xlsxwriter
def write_check(features):
    workbook = xlsxwriter.Workbook('test.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    for key in features:
        worksheet.write(row,col, str(key))
        row+=1
        for item in features[key]:
            worksheet.write(row, col, item)
            row+=1
        row=0
        col+=1

    workbook.close()

# In[]:
check = check_feature(['Bcells', 'CD56.CD16.NKcells', 'CD8.Tcells', 'intMCs', 'M.MDSC', 'pDCs', 'CD16.CD56.NKcells_STAT1_IFNa100', 'CD4.Tcells_STAT1_IFNa100', 'CD8.Tcells_naive_STAT1_IFNa100', 'ncMCs_STAT1_IFNa100', 'Tbet.CD4.Tcells_naive_STAT1_IFNa100', 'Tbet.CD8.Tcells_naive_STAT1_IFNa100', 'TCRgd.Tcells_STAT1_IFNa100', 'CD45RA.Tregs_STAT3_IFNa100.1', 'CD56.CD16.NKcells_STAT3_IFNa100', 'CD8.Tcells_STAT3_IFNa100', 'M.MDSC_STAT3_IFNa100', 'mDCs_STAT3_IFNa100', 'Tregs_STAT3_IFNa100', 'CD8.Tcells_naive_STAT5_IFNa100', 'ncMCs_STAT5_IFNa100', 'Tbet.CD4.Tcells_mem_STAT5_IFNa100', 'TCRgd.Tcells_STAT5_IFNa100', 'CD45RA.Tregs_ERK_IL100', 'cMCs_ERK_IL100', 'M.MDSC_ERK_IL100', 'Tbet.CD4.Tcells_mem_ERK_IL100', 'Tbet.CD8.Tcells_mem_ERK_IL100', 'TCRgd.Tcells_ERK_IL100', 'CD45RA.Tregs_STAT1_IL100', 'CD56.CD16.NKcells_STAT1_IL100', 'intMCs_STAT1_IL100', 'Tbet.CD4.Tcells_mem_STAT1_IL100', 'Tbet.CD8.Tcells_naive_STAT1_IL100', 'Bcells_STAT3_IL100', 'CD4.Tcells_naive_STAT3_IL100', 'CD45RA.Tregs_STAT3_IL100', 'CD8.Tcells_naive_STAT3_IL100', 'CD8.Tcells_STAT3_IL100', 'Gr_STAT3_IL100', 'Tbet.CD8.Tcells_mem_STAT3_IL100', 'TCRgd.Tcells_STAT3_IL100', 'CD4.Tcells_mem_STAT5_IL100', 'CD45RA.Tregs_STAT5_IL100', 'CD8.Tcells_naive_STAT5_IL100', 'cMCs_STAT5_IL100', 'Gr_STAT5_IL100', 'Tbet.CD8.Tcells_naive_STAT5_IL100', 'CD56.CD16.NKcells_CREB_LPS100', 'cMCs_CREB_LPS100', 'Gr_CREB_LPS100', 'CD45RA.Tregs_ERK_LPS100', 'intMCs_ERK_LPS100', 'ncMCs_ERK_LPS100', 'M.MDSC_IkB_LPS100', 'mDCs_IkB_LPS100', 'Tregs_IkB_LPS100', 'CD16.CD56.NKcells_MAPKAPK2_LPS100', 'Gr_MAPKAPK2_LPS100', 'CD56.CD16.NKcells_NFkB_LPS100', 'CD7.NKcells_NFkB_LPS100', 'Gr_NFkB_LPS100', 'intMCs_NFkB_LPS100', 'ncMCs_NFkB_LPS100', 'CD45RA.Tregs_p38_LPS100.1', 'CD56.CD16.NKcells_p38_LPS100',
              'M.MDSC_p38_LPS100', 'mDCs_p38_LPS100', 'Tregs_p38_LPS100', 'CD16.CD56.NKcells_S6_LPS100', 'Gr_S6_LPS100', 'CD4.Tcells_naive_CREB_Unstim', 'CD45RA.Tregs_CREB_Unstim', 'CD56.CD16.NKcells_CREB_Unstim', 'cMCs_CREB_Unstim', 'intMCs_CREB_Unstim', 'Tregs_CREB_Unstim', 'M.MDSC_ERK_Unstim', 'ncMCs_ERK_Unstim', 'Tbet.CD4.Tcells_mem_ERK_Unstim', 'CD45RA.Tregs_IkB_Unstim', 'CD56.CD16.NKcells_IkB_Unstim', 'CD8.Tcells_IkB_Unstim', 'cMCs_IkB_Unstim', 'Gr_IkB_Unstim', 'Tbet.CD4.Tcells_naive_IkB_Unstim', 'CD4.Tcells_naive_MAPKAPK2_Unstim', 'CD56.CD16.NKcells_MAPKAPK2_Unstim', 'CD8.Tcells_naive_MAPKAPK2_Unstim', 'CD8.Tcells_MAPKAPK2_Unstim', 'Gr_MAPKAPK2_Unstim', 'mDCs_MAPKAPK2_Unstim', 'Tbet.CD8.Tcells_mem_MAPKAPK2_Unstim', 'Bcells_NFkB_Unstim', 'CD4.Tcells_mem_NFkB_Unstim', 'CD4.Tcells_naive_NFkB_Unstim', 'CD8.Tcells_naive_NFkB_Unstim', 'intMCs_NFkB_Unstim', 'Tbet.CD4.Tcells_mem_NFkB_Unstim', 'Tbet.CD8.Tcells_naive_NFkB_Unstim', 'CD16.CD56.NKcells_p38_Unstim', 'CD7.NKcells_p38_Unstim', 'CD8.Tcells_mem_p38_Unstim', 'CD8.Tcells_naive_p38_Unstim', 'pDCs_p38_Unstim', 'Tbet.CD8.Tcells_mem_p38_Unstim', 'CD4.Tcells_naive_S6_Unstim', 'CD56.CD16.NKcells_S6_Unstim', 'CD8.Tcells_naive_S6_Unstim', 'M.MDSC_S6_Unstim', 'ncMCs_S6_Unstim', 'Tbet.CD8.Tcells_mem_S6_Unstim', 'Bcells_STAT1_Unstim', 'CD45RA.Tregs_STAT1_Unstim', 'CD7.NKcells_STAT1_Unstim', 'mDCs_STAT1_Unstim', 'CD56.CD16.NKcells_STAT3_Unstim', 'intMCs_STAT3_Unstim', 'pDCs_STAT3_Unstim', 'TCRgd.Tcells_STAT3_Unstim', 'Bcells_STAT5_Unstim', 'CD16.CD56.NKcells_STAT5_Unstim', 'CD4.Tcells_STAT5_Unstim', 'CD45RA.Tregs_STAT5_Unstim', 'CD8.Tcells_naive_STAT5_Unstim', 'mDCs_STAT5_Unstim', 'Tbet.CD4.Tcells_naive_STAT5_Unstim', 'Tbet.CD8.Tcells_naive_STAT5_Unstim'])
write_check(check)

# In[26]:


def main():
    features = read_feature()
    f_new = open("best.csv", "w")
    best_file = csv.writer(f_new)
    best_file.writerow(["generation", "fitness", "features",
                       "number of features", "binary_vector"])

    data, label = read_data()

    population = first_population(data, label)
    # *********************************

    # for i in range(len(population)):
    # print(population[i][1])

    generation_fitness = []
    best_number = 0
    fit_best = 1000

    for g in range(10):
        population.sort(key=lambda tup: tup[1])
        print(g, population[0][1])
        temp = population[0][1]
        if temp < fit_best:
            fit_best = temp
            best_number = 0
        else:
            best_number += 1
            if best_number > size_pop:
                break

        p = population[0]
        name_gene = []
        for i in range(len(p[0])):
            if p[0][i] == 1:
                name_gene.append(features[i])

        best_file.writerow([g+1, p[1], name_gene, len(name_gene), p[0]])

        generation_fitness.append(p[1])
        # print("generation",generation_fitness,"gstop")

        new_population = []

        new_population = cross(population, data, label)
        #print(f"{g}/cross/{[i[1] for i in new_population]}")
        for i in range(N):
            new_population.append(population[i])

        population = mutant(new_population, data, label)
        #print(f"{g}/mutation/{[i[1] for i in population]}")

    # print(best_features)
    plt.plot(list(range(1, g+1)), generation_fitness)
    plt.savefig("data.png")
    plt.show()

    f_new.close()
    
    best_vector = population[0][0]
    best_features = []
    for i in range(len(best_vector)):
        if best_vector[i] == 1:
            best_features.append(features[i])
    check = check_feature(best_features)
    print(check)
    write_check(check)
    return population


# In[27]:
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:
