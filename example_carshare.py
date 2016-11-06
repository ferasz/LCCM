import lccm
import numpy as np
import pandas as pd
import pylogit
import warnings
from collections import OrderedDict

# Load the data file
inputFilePath = 'C:/Users/Feras/Desktop/LCCM Package/'
inputFileName = 'TrainingData.txt'

print '\nReading %s' %inputFileName 
data = np.loadtxt(open(inputFilePath + inputFileName, 'rb'), delimiter='\t')

# Convert to a pandas dataframe
df = pd.DataFrame(data, columns=['indID', 'altID', 'obsID', 'choice', 'zipInd',
        'hhIncome', 'male', 'adopters', 'stationDummy', 'googleDummy', 'accessibility'])

# Clean up and scale variables as needed, create interactions
df['hhIncome'] = df.hhIncome / 1000

df['v_accessibility'] =  df.accessibility
df['v_adopters'] =  df.adopters
df['v_google_dummy'] = df.googleDummy


# Class membership model 
n_classes = 3

class_membership_spec = ['intercept', 'hhIncome', 'male']
class_membership_labels = ['Class-specific constant', 'Monthly Income (1000s $)', 'male' ]



# Class-specific choice model
class_specific_specs = [
	OrderedDict([
		('intercept', [1]), 
		('v_accessibility', [1]), 
		('v_google_dummy', [1]) ]),
	OrderedDict([
		('intercept', [1]),
		('v_accessibility', [1]),
		('v_adopters', [1]),
		('v_google_dummy', [1]) ]),
	OrderedDict([
		('intercept', [1]) ])
]


class_specific_labels = [
	OrderedDict([('ASC', 'ASC (Adopt)'),('Accessibility','Accessibility Adopt'), ('Google Employee', 'Google Employee Adopt')]),
	OrderedDict([('ASC', 'ASC (Adopt)'), ('Accessibility','Accessibility Adopt'), ('Cumulative Adopters (t-1)','Cumulative Adopters (t-1) Adopt'), ('Google Employee','Google Employee Adopt')]),
	OrderedDict([('ASC','ASC (CarShare)')])
]


# Accounting for weights - choice-based sampling (Moshe & Lerman)
# Weighted Exogenous Sample Maximum Likelihood (WESML)
weightAdopters = 0.0003853/0.404
weightNonAdopters = 0.9996147/0.596    
indWeightsA = np.repeat(weightAdopters, 300)
indWeightsNA = np.repeat(weightNonAdopters,201 )        
indWeights = np.hstack((indWeightsA,indWeightsNA))  

# parameter estimates starting value  
paramClassMem = np.zeros(len(class_membership_spec)*(n_classes -1))
paramClassSpec = []
for s in range(0, n_classes):
    paramClassSpec.append(np.array([-1,0,0]))    
    paramClassSpec.append(np.array([-2,0,0,0]))
    paramClassSpec.append(np.array([-15]))

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
                  indWeights = indWeights,
                  outputFilePath = inputFilePath,
                  paramClassMem = paramClassMem,
                  paramClassSpec = paramClassSpec)


