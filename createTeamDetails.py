#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:20:50 2019

@author: jaisi8631
"""
# -------------------------
# IMPORT NECESSARY MODULES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data  
regseason = pd.read_csv('data/RegularSeasonDetailedResults.csv')
postseason = pd.read_csv('data/NCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)
teams = pd.read_csv('data/TeamConferences.csv')

start = (teams.Season.values == 2003).argmax()
teams = teams.iloc[start:]

# set variables of necessary data
seasonIndex = games.columns.get_loc('Season')
WTeamIDIndex = games.columns.get_loc('WTeamID')
LTeamIDIndex = games.columns.get_loc('LTeamID')
WTeamMetrics = ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM',
                'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
WTeamIndexes = []
LTeamMetrics = ['LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
                'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
LTeamIndexes = []

# store index of necessary data within dataset
for i in range(0, len(WTeamMetrics)):
    index = games.columns.get_loc(WTeamMetrics[i])
    WTeamIndexes.append(index)

for i in range(0, len(LTeamMetrics)):
    index = games.columns.get_loc(LTeamMetrics[i])
    LTeamIndexes.append(index)


# -----------------------------
# DATASET SKELETON PREPARATION
# -----------------------------
cols = 5 + len(WTeamMetrics)
rows = teams.shape[0]

dataset = np.zeros(shape = (rows, cols))
dataset = dataset.astype(float)
teamDetails = teams.iloc[:, 0:2].values
dataset[:, 0:2] = teamDetails

rawData = games.iloc[:, :].values
index = games.columns.get_loc("WLoc")
for i in range(0, rawData.shape[0]):
    rawData[i][index] = 0
rawData = rawData.astype(float)


# -------------------------
# DATA PROCESSING
# -------------------------
for i in range(0, rawData.shape[0]):
    
    # output to console
    print("Game #: ", i)
    
    # get descriptive data of game
    thisWTeamID = rawData[i][WTeamIDIndex]
    thisLTeamID = rawData[i][LTeamIDIndex]
    thisSeason = rawData[i][seasonIndex]
    
    # initialize variables
    datasetWTeam = -1
    datasetLTeam = -1
    
    # locate rows of winning and losing teams
    for j in range(0, dataset.shape[0]):
        if(dataset[j][0] == thisSeason):
            if(dataset[j][1] == thisWTeamID):
                datasetWTeam = j
            elif(dataset[j][1] == thisLTeamID):
                datasetLTeam = j
    
    # add a win, a game played and the necessary data to winning team         
    dataset[datasetWTeam][2] += 1
    dataset[datasetWTeam][-1] += 1
    k = 0
    for j in WTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1
    
    # add a loss, a game played and the necessary data to losing team  
    dataset[datasetLTeam][3] += 1
    dataset[datasetLTeam][-1] += 1
    k = 0
    for j in LTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1

# convert totals to averages for statistics       
for i in range(0, dataset.shape[0]):
    games = dataset[i][dataset.shape[1] - 1]
    for j in range(2, dataset.shape[1] - 1):
        if(games != 0): 
            val = float(dataset[i][j]) / float(games)
            val = round(val, 3)
            dataset[i][j] = val


# -------------------------
# DATA STORAGE
# -------------------------    
# save entire dataset to disk
np.savetxt("data/all_teamData.csv", dataset, delimiter = ",")
print("All Team Dataset creation complete.")
print()
print()

# locate index where 2019 data begins
found = -1
i = 0
while(found == -1):
    if(dataset[i][0] == 2019):
        found = i
    i += 1
   
# store subset dataset containing only data from 2019 
curr_year = dataset[found:, :]
np.savetxt("data/2019_teamData.csv", curr_year, delimiter = ",")

# output feedback
print()
print()
print("2019 Team Dataset creation complete.")
print()
print()