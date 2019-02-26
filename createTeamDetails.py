#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:20:50 2019

@author: jaisi8631
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


regseason = pd.read_csv('RegularSeasonDetailedResults.csv')
postseason = pd.read_csv('NCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)
teams = pd.read_csv('TeamConferences.csv')

start = (teams.Season.values == 2003).argmax()
teams = teams.iloc[start:]

seasonIndex = games.columns.get_loc('Season')
WTeamIDIndex = games.columns.get_loc('WTeamID')
LTeamIDIndex = games.columns.get_loc('LTeamID')
WTeamMetrics = ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM',
                'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
WTeamIndexes = []
LTeamMetrics = ['LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
                'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
LTeamIndexes = []

for i in range(0, len(WTeamMetrics)):
    index = games.columns.get_loc(WTeamMetrics[i])
    WTeamIndexes.append(index)

for i in range(0, len(LTeamMetrics)):
    index = games.columns.get_loc(LTeamMetrics[i])
    LTeamIndexes.append(index)


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


for i in range(0, rawData.shape[0]):
    
    print("Game #: ", i)
    
    thisWTeamID = rawData[i][WTeamIDIndex]
    thisLTeamID = rawData[i][LTeamIDIndex]
    thisSeason = rawData[i][seasonIndex]
    
    datasetWTeam = -1
    datasetLTeam = -1
    
    for j in range(0, dataset.shape[0]):
        if(dataset[j][0] == thisSeason):
            if(dataset[j][1] == thisWTeamID):
                datasetWTeam = j
            elif(dataset[j][1] == thisLTeamID):
                datasetLTeam = j
             
    dataset[datasetWTeam][2] += 1
    dataset[datasetWTeam][dataset.shape[1] - 1] += 1
    k = 0
    for j in WTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1
    
    dataset[datasetLTeam][3] += 1
    dataset[datasetWTeam][dataset.shape[1] - 1] += 1
    k = 0
    for j in LTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1
        
for i in range(0, dataset.shape[0]):
    games = dataset[i][dataset.shape[1] - 1]
    for j in range(2, dataset.shape[1] - 1):
        if(games != 0): 
            dataset[i][j] = float(dataset[i][j]) / float(games)
    

np.savetxt("teamdata.csv", dataset, delimiter=",")