import sys
import os
import pandas as pd

df= pd.read_csv('main/data/ingested/train.csv')

x = 'Depression'

target = df[[x]]

print(target)

