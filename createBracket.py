from bracketeer import build_bracket 

m = build_bracket(outputPath = 'bracket.png',
	teamsPath = 'data/Teams.csv',
	seedsPath = 'data/NCAATourneySeeds.csv',
	submissionPath = 'submission.csv',
	slotsPath = 'data/NCAATourneySlots.csv',
	year = 2019)