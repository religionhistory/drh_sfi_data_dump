# DRH SFI data dump (2023/06/30)


## Data 

Data were downloaded from the [Database of Religious History (DRH)](https://religiondatabase.org/landing/) on 30<sup>th</sup> June 2023 for use at the SFI Working Group "The Database of Religious History & Cultural Evolution". The included variables in drh.csv are:

  - Standardized Question ID: Question ID that has been standardized between poll types.
  - Standardized Question: Question that has been standardized between poll types.
  - Standardized Parent question: Parent question which has been standardized between poll types, which must have been answered a certain way for the question to have been asked.
  - Poll: The poll answered by the entry. This is either Religious Group, Religious Place or Religious Text, along with the version of the poll answered.
  - Original question ID: Original question ID, which is poll specific. 
  - Original Question: Question, which is worded in a poll specific manner. 
  - Original Parent question: The original parent question, which must have been answered a certain way for the question to have been asked. 
  - Answers: The answer to the question. This can be categorical, numeric or text.
  - Answer Values: A numeric value representing the answer. For the most common answer set of Yes/No/Field doesn't know and I don't know, Yes = 1, No = 0, Field doesn't know and I don't know = -1. 
  - Note: Commentary about the reason for the coding decision. 
  - Parent answer: Answer to parent question. 
  - Parent answer value: A numeric answer representing the answer to the parent question.
  - Branching question: The societal segment that the answer corresponds to. The possible segments are: Religious Specialists, Elite and Non-elite (common people, general populace)
  - Entry name: Name of the entry.
  - Entry ID: ID of the entry.
  - Entry description: Description of the entry.
  - Date range: Date range of the answer, as a string in the format of year(BCE/CE) - year (BCE/CE).
  - start_year: Start year of the answer as a numeric value, with BCE dates being represented as negative numbers.
  - end_year: End year of the answer as a numeric value, with BCE dates being represented as negative numbers.
  - Entry source: Data source of the entry.
  - Entry tags: Entry tags corresponding to the entry. For Religious Group entries these are religious group tags, for Religious Place entries these are place and religious group tags and for Religious Text entries these are text and religious group tags.
  - Region name: Name of the region corresponding to the answer.
  - Region ID: ID of the region corresponding to the answer. This corresponds to the KML file name. 
  - Region description: Description of the region corresponding to the answer.
  - Region tags: Region tags corresponding to the entry. 
  - Expert: Expert who created the entry.