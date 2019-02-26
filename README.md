# March Madness 2019
A Machine Learning project to predict the results of the NCAA Men's March Madness Basketball tournament.

Datasets used are derived from Google Cloud and NCAA's Kaggle Competition on March Madness: https://www.kaggle.com/c/mens-machine-learning-competition-2019


To execute the project, complete the following steps:
1. Create dataset that holds details for each team for every season.
~~~~
python create-team-details.py
~~~~~~~~ 

2. Create dataset that contains details about every past matchup.
~~~~
python create-training-dataset.py
~~~~~~~~ 

3. Create predictive models that can be used to predict the outcomes of future fixtures.
~~~~
python create-prediction-model.py
~~~~~~~~ 


The features in an input instance for this model are:	
* 0: Team 1 Home or Away (1: Home, 0: Neutral, -1: Away)
* 1: Team 1 Points per game - Team 2 Points per game
* 2: Team 1 Points Allowed per game - Team 2 Points Allowed per game
* 3: Team 1 Field Goals Made per game - Team 2 Field Goals Made per game
* 4: Team 1 Field Goals Attempted per game - Team 2 Field Goals Attempted per game
* 5: Team 1 3-Pointers Made per game - Team 2 3-Pointers Attempted per game
* 6: Team 1 Free-Throws Made per game - Team 2 Free-Throws Attempted per game
* 7: Team 1 Offensive Rebounds per game - Team 2 Offensive Rebounds per game
* 8: Team 1 Defensive Rebounds per game - Team 2 Defensive Rebounds per game
* 9: Team 1 Assists per game - Team 2 Assists per game
* 10: Team 1 Turnovers per game - Team 2 Turnovers per game
* 11: Team 1 Steals per game - Team 2 Steals per game
* 12: Team 1 Blocks per game - Team 2 Blocks per game
* 13: Team 1 Personal Fouls per game - Team 2 Personal Fouls per game

The output for an input instance for this model is:
* Whether or not Team 1 wins (0: Team 1 Loses, 1: Team 1 Wins)
