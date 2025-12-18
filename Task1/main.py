import pandas as pd
from yargy import Parser, rule, or_,and_
from yargy.predicates import gram, is_capitalized,eq
from yargy.predicates import normalized, type as t
from yargy.interpretation import fact
from yargy.pipelines import morph_pipeline

df = pd.read_csv(
    "newsTest.txt",
    sep="\t",
    header=None,
    names=["tag", "title", "text"],
    encoding="utf-8"
)

EntryFact = fact(
    'Entry',
    ['name', 'birth_date', 'birth_place']
)

#region NAME_RULE
SURNAME = and_(is_capitalized(), gram('Surn'))
NAME = and_(gram('Name'), is_capitalized())
PATR = and_(gram('Patr'), is_capitalized())

NAME_RULE = or_(
    rule(NAME, SURNAME),
    rule(NAME, SURNAME),
    rule(SURNAME, NAME),
    rule(NAME, PATR),
    rule(SURNAME, NAME, PATR),
)
#endregion

#region BIRTH_PLACE_RULE
BIRTH_PLACE_RULE = rule(
    or_(eq('в'), eq('на'), eq('при'), eq('около'), eq('у')),
    gram('Geox').interpretation(EntryFact.birth_place)
)
#endregion

#region BIRTH_DATE_RULE
INT = t('INT')

MONTH_RULE=morph_pipeline(
    {
    'январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь'})

full_date_rule = (rule(INT,MONTH_RULE,INT)
                  .interpretation(EntryFact.birth_date))

year_only_rule = INT.interpretation(EntryFact.birth_date)

# родился 15 сентября 1945
# родился в 1945
BIRTH_DATE_RULE = or_(
    rule(full_date_rule,normalized('год').optional()),
    rule('в',year_only_rule,normalized('год').optional()),
)
#endregion

#region Main
TASK_MAIN_RULE = rule(
    NAME_RULE.interpretation(EntryFact.name),
    normalized('родиться'),
    or_(
        rule(BIRTH_DATE_RULE,BIRTH_PLACE_RULE.optional()),
        rule(BIRTH_PLACE_RULE,BIRTH_DATE_RULE.optional()),
    )
).interpretation(EntryFact)

parser = Parser(TASK_MAIN_RULE)

for text in df["text"]:
    for match in parser.findall(text):
        print(match.fact)

#endregion
