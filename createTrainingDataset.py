#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:38:04 2019

@author: jaisi8631
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

teamdetails = pd.read_csv('teamdata.csv', header=None)
rawTeamData = teamdetails.iloc[:, :].values

regseason = pd.read_csv('RegularSeasonDetailedResults.csv')
postseason = pd.read_csv('NCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)

cols = 23
rows = games.shape[0]
dataset = np.zeros(shape = (rows, cols))
dataset = dataset.astype(float)

for i in range(0, games.shape[0]):
    
    print("Game #: ", i)
    
    season = games.iloc[i]['Season']
    WTeamID = games.iloc[i]['WTeamID']
    LTeamID = games.iloc[i]['LTeamID']
    
    x = games.iloc[i]['WLoc']
    loc = 0
    if(x == 'N'):
        loc = 0
    elif(x == 'H'):
        loc = 1
    elif(x == 'A'):
        loc = -1
    
    if(season == 2019):
        break
    
    values = np.array([i+1, season])
    values = values.astype(float)
          
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == WTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                WTeamData = rawTeamData[j, 2:-1]
        j += 1
        
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == LTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                LTeamData = rawTeamData[j, 2:-1]
        j += 1
    
    winner = -1
    difference = np.subtract(WTeamData, LTeamData)
    x = random.uniform(0, 1)
    if(x > 0.5):
        winner = 1
        values = np.append(values, WTeamID)
        values = np.append(values, LTeamID)
        values = np.append(values, loc)
        
    else:
        winner = 0
        loc = loc * -1
        difference = difference * -1
        values = np.append(values, LTeamID)
        values = np.append(values, WTeamID)
        values = np.append(values, loc)
    
    difference = np.append(difference, winner)
    
    instance = np.concatenate([values, difference])
    
    dataset[i] = instance

np.savetxt("dataset.csv", dataset, delimiter=",")