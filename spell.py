
'''
from spellchecker import SpellChecker

spell = SpellChecker()

# find those words that may be misspelled
misspelled = spell.unknown(['somethingis', 'isacode', 'hapenning', 'here'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    #print(spell.candidates(word))



import splitter

splitter.split('artfactory')

'''
import pandas as pd
from spellchecker import SpellChecker

eng = pd.Series(['EmpName', 'EMP_NAME', 'EMP.NAME', 'EMPName', 'CUSTOMIR', 'TIER187CAST', 'MultipleTIMESTAMPinTABLE', 'USD$'])
eng = eng.str.lower()
eng = eng.str.split()
spell = SpellChecker()
def msp(x):
    return spell.unknown(x)
eng.apply(msp)

print (eng)