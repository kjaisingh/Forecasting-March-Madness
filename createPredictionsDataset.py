#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:17:44 2019

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
submissions = pd.read_csv('SampleSubmissionStage2.csv')
data = pd.read_csv('data/2019_teamData.csv', header=None)
matchups = submissions.iloc[:, :].values
rawTeamData = data.iloc[:, :].values


# -----------------------------
# DATASET SKELETON PREPARATION
# -----------------------------
rows = matchups.shape[0]
cols = 22
dataset = np.zeros(shape = (rows, cols))


# -------------------------
# DATA PROCESSING
# -------------------------
for i in range(0, dataset.shape[0]):
    
    # split game identifier by specified delimiter
    string = matchups[i][0]
    keys = string.split("_")
    
    # assign variables that correspond to each piece of information
    year = int(keys[0])
    team1ID = int(keys[1])
    team2ID = int(keys[2])
    
    # create outline of instance in the dataset
    instance = np.array([i + 1, year, team1ID, team2ID, 0])
    
    # locate index of team 1's season data
    found = -1
    j = 0
    while(found == -1 and j < rawTeamData.shape[0]):
        if(rawTeamData[j][1] == team1ID):
            found = j
            team1Data = rawTeamData[j, 2:-1]
        j += 1
        
    # locate index of team 2's season data
    found = -1
    j = 0
    while(found == -1 and j < rawTeamData.shape[0]):
        if(rawTeamData[j][1] == team2ID):
            found = j
            team2Data = rawTeamData[j, 2:-1]
        j += 1

    # calculate the difference between team 1 and team 2's season data
    difference = np.subtract(team1Data, team2Data)
    
    # merge the difference with the basic matchup data
    instance = np.concatenate([instance, difference])
    
    # add the instance to the dataset
    dataset[i] = instance


# -------------------------
# DATA STORAGE
# ------------------------- 
np.savetxt("data/2019_dataset.csv", dataset, delimiter=",")