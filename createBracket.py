from bracketeer import buildbracket 

m = buildbracket(outputPath='output.png',
	teamsPath='data/Teams.csv',
	seedsPath='data/NCAATourneySeeds.csv',
	submissionPath='SampleSubmissionStage2.csv',
	slotsPath='data/NCAATourneySlots.csv',
	year=2019)