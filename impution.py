
from pyfrechet_k.metric_spaces import MetricData, Wasserstein1D
from pyfrechet_k.metrics import mse
from pyfrechet_k.regression.frechet_regression import GlobalFrechet
from adaboostval import AdaBoost
import pandas
import numpy as np
import matplotlib. pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error

hours = pandas.read_csv('./bike+sharing+dataset/hour.csv')

print(hours.info())
print(hours.loc[27:32])
listofvars = ['dteday','yr', 'mnth', 'season','holiday','weekday','workingday']
j = 1
while j <48:
    for i in range(len(hours) - 1):
        if hours.loc[i+1]["hr"] - hours.loc[i]["hr"] > 1: # missing hours, same day
       
        
            hours.loc[i + 0.5] = [np.nan for k in hours.keys()]
            hours["cnt"][i + 0.5] = 0
            hours["hr"][i + 0.5] = hours["hr"][i] + 1
          
            for vars in listofvars:
                if hours.loc[i]["dteday"] == hours.loc[i+1]["dteday"] and hours.loc[i]["hr"] < 23:
                    r_dteday = hours.loc[i][vars]
                else:
                    r_dteday = hours.loc[i+1][vars]

               
                hours.loc[i + 0.5, vars] = r_dteday

        if (hours.loc[i]["dteday"] != hours.loc[i+1]["dteday"]\
        and hours.loc[i+1]["hr"] > 0): # if a day doesn't start with 0

        
        
            
            hours.loc[i + 0.5] = [np.nan for k in hours.keys()]
            hours["cnt"][i + 0.5] = 0
            hours["hr"][i + 0.5] =  hours.loc[i+1]["hr"]-1

           
            for vars in listofvars:
                if hours.loc[i]["dteday"] == hours.loc[i+1]["dteday"] and hours.loc[i]["hr"] < 23:
                    r_dteday = hours.loc[i][vars]
                else:
                    r_dteday = hours.loc[i+1][vars]

             
                hours.loc[i + 0.5, vars] = r_dteday

        # Missing values over two days (statement overlaps with previous, but isn't active if solved previously)
        if (hours.loc[i+1]["hr"] - hours.loc[i]["hr"] < 1 \
        
        and hours.loc[i]["hr"] != 23):
            
       
            
            
            hours.loc[i + 0.5] = [np.nan for k in hours.keys()]
            hours["cnt"][i + 0.5] = 0
            hours["hr"][i + 0.5] = hours["hr"][i] + 1
           
            for vars in listofvars:
                r_dteday = hours.loc[i][vars]

            
                hours.loc[i + 0.5, vars] = r_dteday
        

    hours = hours.sort_index().reset_index(drop=True)
    j=j+1


hours = hours.sort_index().reset_index(drop=True)


print("   value counts    ")
print(hours['dteday'].value_counts()[hours['dteday'].value_counts()<24])


print("   value counts 2   ")
print(hours['dteday'].value_counts()[hours['dteday'].value_counts()>24])



hours.interpolate(method = 'linear', inplace = True, limit_direction ='forward', axis = 0)


hours.to_csv('hoursimp.csv')  





print(np.datetime64(hours["dteday"][27])-np.datetime64(hours["dteday"][0])<2)