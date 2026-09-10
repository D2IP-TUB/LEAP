"""TableQA answer-generation prompts adapted from AutoPrep."""

NL2CODE_DEMO = """You are an agent that generates Python code for table question answering.
The table is stored in a pandas DataFrame named `df`.
Store the final numerical value, string, or list in `result`.

Title: Portugal in the Eurovision Song Contest 1979
/*
row_id draw artist
0 1 gonzaga coutinho
1 2 pedro osório s.a.r.l.
2 3 concha
*/
Q: who was the last draw?
Code: ```result = df.loc[df['draw'] == df['draw'].max(), 'artist'].values[0]```

Title: 2007 New Orleans Saints season
/*
row_id week opponent game site result/score
0 1 indianapolis colts away l 41 - 10
1 2 tampa bay buccaneers away l 31 - 14
2 3 tennessee titans home l 31 - 14
*/
Q: what number of games were lost at home?
Code: ```home_losses = df[df['game site'] != 'away']
result = len(home_losses[home_losses['result/score'].str.startswith('l')])```

Title: Płock Governorate
/*
row_id language males females
0 polish 216794 230891
1 yiddish 24538 26677
2 german 17409 18522
*/
Q: how many male and female german speakers are there?
Code: ```male = df.loc[df['language'] == 'german', 'males'].values[0]
female = df.loc[df['language'] == 'german', 'females'].values[0]
result = [male, female]```"""

NL2CODE_QUERY = """Please complete the prompt following the format above.
/*
{table}
*/
Q: {question}
Output ```your_code_here``` with no other text.
Code:"""

END2ENDER_DEMO = """Here is the table to answer this question. Answer the question.
/*
col : Rank | Cyclist | Team | Points
row 1 : 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 40
row 2 : 2 | Alexandr Kolobnev (RUS) | Team CSC | 30
row 3 : 3 | Davide Rebellin (ITA) | Gerolsteiner | 25
row 4 : 4 | Paolo Bettini (ITA) | Quick Step | 20
row 5 : 5 | Franco Pellizotti (ITA) | Liquigas | 15
*/
Question: which country had the most cyclists finish within the top 5?
The answer is: Italy.

Here is the table to answer this question. Answer the question.
/*
col : Rank | Cyclist | Points
row 1 : 1 | Alejandro Valverde | 40
row 2 : 2 | Alexandr Kolobnev | 30
row 3 : 3 | Davide Rebellin | 7
row 4 : 4 | Paolo Bettini | 5
*/
Question: how many players got less than 10 points?
The answer is: 2."""

END2ENDER_QUERY = """Here is the table to answer this question. Answer the question.
/*
{table}
*/
Question: {question}
The answer is:"""

COT_END2ENDER_DEMO = """Read the table below to answer the questions.

col: Rank | Cyclist | Team | Points
row1: 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 40
row2: 2 | Alexandr Kolobnev (RUS) | Team CSC | 30
row3: 3 | Davide Rebellin (ITA) | Gerolsteiner | 25
row4: 4 | Paolo Bettini (ITA) | Quick Step | 7
row5: 5 | Franco Pellizotti (ITA) | Liquigas | 5

Question: which country had the most cyclists finish within the top 5?
Explanation: ITA occurs three times, more than any other country. Therefore, the answer is Italy.

Question: how many players got less than 10 points?
Explanation: Paolo Bettini and Franco Pellizotti got less than 10 points. Therefore, the answer is 2."""

COT_END2ENDER_QUERY = """Read the table below to answer the question.
/*
{table}
*/
Question: {question}
Answer the question based on the table with the format above.
Explanation:"""
