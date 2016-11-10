import lccm
import numpy as np
import pandas as pd
import pylogit
import warnings
from collections import OrderedDict

# Load the data file
inputFilePath = 'C:/Users/Feras/Desktop/LCCM Package/'
inputFileName = 'Santiago.csv'

print '\nReading %s' %inputFileName 
df = pd.read_csv(open(inputFilePath + inputFileName, 'rb'))
df = df.iloc[:,0:13]

# defining the column names of dataframe
df.columns=['indID', 'obsID', 'choice', 'altID', 'waveID', 'traveltime','AccEgressTime',
                                 'waitTime', 'Num Transfers', 'travelCost', 'workHours', 'income','male']



# Class membership model 
df['income'] = df.income / 1000
n_classes = 2

class_membership_spec = ['intercept', 'income', 'male']
class_membership_labels = ['Class-specific constant', 'Monthly Income (1000s $)', 'male' ]



# Class-specific choice model
class_specific_specs = [
	OrderedDict([
		('intercept', [2,3,4,5,6,7]), 
		('traveltime', [[1],[2,3,4,5,6,7]]), 
		('travelCost', [[2,3,7],[1,4,5,6]]) 
   ]),
	OrderedDict([
		('intercept', [2,3,4,5,6,7]),     
		('traveltime', [[1,2,3,4,5,6,7]]), 
		('travelCost', [[1,2,3,4,5,6,7]]) 
   ])
]


class_specific_labels = [
	OrderedDict([('ASC', ['ASC(Metro)', 'ASC(Bus)', 'ASC(Walk)', 'ASC(Bike)' ,'ASC(AutoMetro)', 'ASC(BusMetro)']),
               ('Travel Time',['Travel Time Car','Travel Time all but car']), ('Travel Cost', ['Travel Cost bus and metro','Travel cost all but bus and metro'])]),
	OrderedDict([('ASC', ['ASC(Metro)', 'ASC(Bus)', 'ASC(Walk)', 'ASC(Bike)', 'ASC(AutoMetro)']),
               ('Travel Time',['Travel Time']), ('Travel Cost', ['Travel Cost'])])
]


# Fit the model

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    lccm.lccm_fit(data = df,
                  ind_id_col = 'indID', 
                  obs_id_col = 'obsID',
                  alt_id_col = 'altID',
                  choice_col = 'choice', 
                  n_classes = n_classes,
                  class_membership_spec = class_membership_spec,
                  class_membership_labels = class_membership_labels,
                  class_specific_specs = class_specific_specs,
                  class_specific_labels = class_specific_labels,
                  #indWeights = indWeights,
                  outputFilePath = inputFilePath)
                  #paramClassMem = paramClassMem)
                  #paramClassSpec = paramClassSpec)


