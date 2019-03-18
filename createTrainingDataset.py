#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:38:04 2019

@author: jaisi8631
"""
# -------------------------
# IMPORT NECESSARY MODULES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data 
teamdetails = pd.read_csv('data/all_teamData.csv', header=None)
rawTeamData = teamdetails.iloc[:, :].values

regseason = pd.read_csv('data/RegularSeasonDetailedResults.csv')
postseason = pd.read_csv('data/NCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)


# -----------------------------
# DATASET SKELETON PREPARATION
# -----------------------------
cols = 23
rows = games.shape[0]
dataset = np.zeros(shape = (rows, cols))
dataset = dataset.astype(float)


# -------------------------
# DATA PROCESSING
# -------------------------
for i in range(0, games.shape[0]):
    
    # output to console
    print("Game #: ", i)
    
    # get descriptive data of game
    season = games.iloc[i]['Season']
    WTeamID = games.iloc[i]['WTeamID']
    LTeamID = games.iloc[i]['LTeamID']
    
    # identify location of game (home, away or neutral)
    x = games.iloc[i]['WLoc']
    loc = 0
    if(x == 'N'):
        loc = 0
    elif(x == 'H'):
        loc = 1
    elif(x == 'A'):
        loc = -1
    
    # convert array to float
    values = np.array([i+1, season])
    values = values.astype(float)
          
    # add raw team data of winning team
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == WTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                WTeamData = rawTeamData[j, 2:-1]
        j += 1
        
    # add raw team data of losing team
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == LTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                LTeamData = rawTeamData[j, 2:-1]
        j += 1
    
    # find difference between winning and losing team statistics
    winner = -1
    difference = np.subtract(WTeamData, LTeamData)
    
    # randomize for which will come first, and modify data accordingly
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
        
    # add winnning team position to data
    difference = np.append(difference, winner)
    difference = np.round(difference, decimals = 3)
    
    # merge descriptive data with team data
    instance = np.concatenate([values, difference])
    
    # add game statistics to dataset
    dataset[i] = instance


# -------------------------
# DATA STORAGE
# ------------------------- 
np.savetxt("data/all_dataset.csv", dataset, delimiter = ",")
print("Training Dataset creation complete.")
print()
print()