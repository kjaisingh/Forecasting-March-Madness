# March Madness Tournament Outcome Prediction

### A Machine Learning system to guide your March Madness bracket.

A Machine Learning project providing a solution to the Google Cloud & NCAA® Machine Learning Competition 2019 (https://www.kaggle.com/c/mens-machine-learning-competition-2019).

The project provides a solution to both Stage 1 and Stage 2 of the Kaggle competition, predicting the winner of historical fixtures and predicting the winners of all possible March Madness 2019 matchups respectively. 

The submission files for these solutions can be found in 'SubmissionStage1.csv' and 'SubmissionStage2.csv' respectively. The project predicts the probability that the first team specified will beat the second team specified for each matchup listed in the Sample Submission file for both Stage 1 and Stage 2.

The project also creates a predicted bracket for the 2019 NCAA Men's March Madness basketball tournament based on the predictions made in Stage 2, and uses the tournament seedings and slots. This bracket can be found in the file 'output.png'.


**Required Dependencies:**
* Numpy
* Pandas
* Matplotlib
* Pickle
* Random
* Ski Kit Learn
* Keras
* Bracketeer


**Execution Instructions:**
1. Create dataset that holds details for each team for every season.
~~~~
python createTeamDetails.py
~~~~~~~~ 

2. Create dataset that contains details about every past matchup, which is used as the training dataset.
~~~~
python createTrainingDataset.py
~~~~~~~~ 

3. Create predictive models that can be used to predict the outcomes of future fixtures.
~~~~
python createPredictionModel.py
~~~~~~~~ 

4. Create dataset that holds the data used by the model for matchups that must be predicted in the Stage 1 submission.
~~~~
python createStage1PredictionsDataset.py
~~~~~~~~ 

5. Use the most accurate model created to create Stage 1 predictions, and write these to a submission file.
~~~~
python createStage1Predictions.py
~~~~~~~~ 

6. Submit the Stage 1 predictions to the Kaggle competition (requires setting up of the Kaggle library and Kaggle API).
~~~~
kaggle competitions submit -c mens-machine-learning-competition-2019 -f SubmissionStage1.csv -m "My Stage 1 submission"
~~~~~~~~ 

7. Create dataset that holds the data used by the model for matchups that must be predicted in the Stage 2 submission.
~~~~
python createStage2PredictionsDataset.py
~~~~~~~~ 

8. Use the most accurate model created to create Stage 2 predictions, and write these to a submission file.
~~~~
python createStage2Predictions.py
~~~~~~~~ 

9. Create a visual representation of the predictions made for the 2019 tournament.
~~~~
python createBracket.py
~~~~~~~~ 

10. Submit the Stage 2 predictions to the Kaggle competition (requires setting up of the Kaggle library and Kaggle API).
~~~~
kaggle competitions submit -c mens-machine-learning-competition-2019 -f SubmissionStage2.csv -m "My Stage 2 submission"
~~~~~~~~ 


**File Details:** \
Other files included or created in this repository include (in order of creation/access):
* *data/RegularSeasonDetailedResults.csv*: Holds data from NCAA Regular Season matchups since 1985.
* *data/NCAATourneyDetailedResults.csv*: Holds data from NCAA March Madness matchups since 2003.
* *data/TeamConferences.csv*: Holds data regarding the team ID's of each team part of the dataset for each year.
* *data/all_teamData.csv*: Holds per-season data for each NCAA team since 2003.
* *data/2019_teamData.csv*: Holds regular season data for 2019 NCAA teams.
* *data/all_dataset.csv*: Holds data for all NCAA matchups since 2003 in a format suitable for use as training data.
* *scaler.save*: Holds the scaler that is used to preprocess data before it is used for predictions.
* *predictor.pkl*: Holds the most accurate classifier created during the training phase.
* *data/stage1_dataset.csv*: Holds data for matchups identified in Stage 1 in a format suitable for making predictions with.
* *data/stage2_dataset.csv*: Holds data for matchups identified in Stage 2 in a format suitable for making predictions with.
* *SampleSubmissionStage1.csv*: Holds details regarding which matchups should be predicted for Stage 1.
* *SampleSubmissionStage2.csv*: Holds details regarding which matchups should be predicted for Stage 2.
* *SubmissionStage1.csv*: Holds the submissions for Stage 1 of the Kaggle competition, as it stores the matchup predictions.
* *SubmissionStage2.csv*: Holds the submissions for Stage 2 of the Kaggle competition, as it stores the matchup predictions.
* *data/Teams.csv*: Holds data regarding the team name of each team based on their team ID. 
* *data/NCAATourneySeeds.csv*: Holds data regarding the NCAA seed of each team.
* *data/NCAATourneySlots.csv*: Holds data regarding the NCAA slot of each team.
* *bracket.png*: Holds a visual representation of the predictions made in the form of a bracket for the 2019 tournament.


**Predictor Details:** \
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

The output for an input instance for this model is the probability that Team 1 wins.


**NCAA 2019 Tournament Predictions:** \
<img src = "bracket.png"/>


**Reflection on the Competition, the Model and the Results:** \
With March Madness 2019 now over, I thought it would be a great time to reflect on my first time entering Google's competition, which saw entries from some of the leading universities and academic teams in the world. So, let's get to it.
* In terms of its simplicity to implement, the model turned out to be decent. It predicted the right winner of each matchup that occurred in this tournament with an accuracy of about 78%, which isn't too bad. Better still, it was able to predict the correct winner of the entire tournament - the Virginia Cavaliers. This indicates that it did, to an extend, understand what was important in winning - in the case of Virginia, it seemed to be great defense and solid three-point shooting (which is probably why the model didn't even predict pre-tournament favourites Duke to make the finals).
* The main flaw that I saw in the model was the fact that it did not take seedings into account. While the seeding would be the basis for most people's predictions in fixtures, as there tends to be a relatively strong correlation between the higher seed and the winning team, my model did not consider the seeding of the two teams in a matchup. This resulted in the model often making extremely risky predictions that were purely stats-based - I believe that the imposition of the seedings of the two teams will reduce the number of these risky predictions. The reason why it was difficult to incorporate seedings into the model was that the large majority of training data was from regular NCAA matches rather than the March Madness Championships. The large majority of teams playing in these NCAA matches, however, do not ever get assigned a seed - this is because they do not qualify for the Championships  itself. To get around this issue, I plan to utilize seedings in future predictive models with one slight tweak - I would assign a seeding of 16, which is the lowest possible seed in the Championships, to all teams that do not end up ever getting a seed. This indicates that they would be the weakest in the pool of teams if they qualified the Championships, which is valid given that they did not even make it.
* Another shortcoming of the model, in my eyes, was that it didn't consider any of the previous matchups between the two teams considered. Head-to-head is usually a pretty good indicative of which team will win, since it is derived from experience. Adding this in future editions shouldn't be too difficult - it'll just require a data point that represents the yearly matchup details between two teams.
* The project largely focused on the data side of the model, so little time was spent on optimizing the model - only a GridSearch was applied for the neural network. Optimization is, however, a clear distinguishing factor between average and great models. In order to improve on this next year, I would have to spend more time reading about about the latest advancements in the domain, and apply techniques that generate improvements which optimize the model. This may, however, require the implementation of techniques from new research papers.
