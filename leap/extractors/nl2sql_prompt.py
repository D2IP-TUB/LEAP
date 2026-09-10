"""Prepared-table NL2SQL prompt adapted from AutoPrep."""

DEMO_NL2SQLER_PREP = """Generate SQL given the question and table to answer the question correctly.

CREATE TABLE w(
    row_id int,
    draw int,
    artist text)
Title: Portugal in the Eurovision Song Contest 1979
/*
row_id draw artist
0 1 gonzaga coutinho
1 2 pedro osório s.a.r.l.
2 3 concha
*/
Q: who was the last draw?
SQL: ```SELECT `artist` FROM w ORDER BY `row_id` DESC LIMIT 1```

CREATE TABLE w(
    row_id int,
    title text,
    artist text)
Title: The Boys (comics)
/*
row_id artist title
0 garth ennis the name of the game
1 carlos ezquerra get some
2 darick robertson good for the soul
*/
Q: what title appears before "the self-preservation society"?
SQL: ```SELECT `title` FROM w WHERE row_id = (SELECT row_id FROM w WHERE `title` = 'the self-preservation society') - 1```

CREATE TABLE w(
    row_id int,
    name text,
    wins int)
Title: Fabrice Santoro
/*
row_id name wins
0 at australian open 22
1 at french open 17
2 at wimbledon 11
*/
Q: did he win more at the australian open or indian wells?
SQL: ```SELECT CASE
WHEN (SELECT wins FROM w WHERE name = 'at australian open') > (SELECT wins FROM w WHERE name = 'at indian wells')
THEN 'australian open' ELSE 'indian wells' END```

CREATE TABLE w(
    row_id int,
    language text,
    males int,
    females int)
Title: Płock Governorate
/*
row_id language males females
0 polish 216794 230891
1 yiddish 24538 26677
2 german 17409 18522
*/
Q: how many male and female german speakers are there?
SQL: ```SELECT `males` + `females` FROM w WHERE `language` = 'german'```

CREATE TABLE w(
    row_id int,
    administrative_area text,
    area_km2 real)
Title: Saint Helena, Ascension and Tristan da Cunha
/*
row_id administrative_area area_km2
0 saint helena 122.0
1 ascension island 91.0
2 tristan da cunha 184.0
*/
Q: is the area of saint helena more than that of nightingale island?
SQL: ```SELECT CASE
WHEN (SELECT area_km2 FROM w WHERE administrative_area = 'saint helena')
  > (SELECT area_km2 FROM w WHERE administrative_area = 'nightingale island')
THEN 'yes' ELSE 'no' END```

CREATE TABLE w(
    row_id int,
    wrestler text,
    date text)
Title: WSL World Heavyweight Championship
/*
row_id wrestler date
0 jonnie stewart 1996-6-6
1 king kong bundy 1999-3-31
2 the patriot 2000-7-29
*/
Q: when did steve corino win his first wsl title?
SQL: ```SELECT `date` FROM w WHERE `wrestler` = 'steve corino' ORDER BY `date` LIMIT 1```"""

QUERY_NL2SQLER_PREP = """{create_table_text}
/*
example rows:
SELECT * FROM w;
{table}
*/
Q: {question}
Output ```your_sql_here``` with no other text.
SQL:"""
