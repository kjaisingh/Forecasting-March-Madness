from bracketeer import build_bracket 

m = build_bracket(outputPath = 'bracket.png',
	teamsPath = 'data/Teams.csv',
	seedsPath = 'data/NCAATourneySeeds.csv',
	submissionPath = 'SubmissionStage2.csv',
	slotsPath = 'data/NCAATourneySlots.csv',
	year = 2019)

print("2019 Bracket creation complete.")
print()
print()