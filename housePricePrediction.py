import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df_data = pd.read_csv("D:\Datasets\Regression\HousePricePrediction\data.csv")
df_output = pd.read_csv("D:\Datasets\Regression\HousePricePrediction\output.csv")

scaler = StandardScaler()
encoder = LabelEncoder()


df_data["date"] = pd.to_datetime(df_data["date"])
df_output["date"] = pd.to_datetime(df_output["date"])

df_data.drop(columns=["date","street","statezip","country"], inplace = True)
df_output.drop(columns=["date","street","statezip","country"], inplace = True)


df_data["city"] = encoder.fit_transform(df_data["city"])
df_output["city"] = encoder.fit_transform(df_output["city"])

frames = [df_data,df_output]

df_New = pd.concat(frames)


df_y = df_New["price"].values

df_New.drop(columns=["price"], inplace = True)

df_x = df_New.values


# df_data_scaled = scaler.fit_transform(df_data[["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","sqft_basement","city","sqft_above","yr_built","yr_renovated"]])

# Splitting the dataset into training and testing

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size= 0.33, random_state= 0)

print("-------------------------------------------------------------------------------------------------------------------------")




model_params = {
    'linear Regression' : {
        'model' : LinearRegression(),
        'parameters' : {
            'fit_intercept' : [True,False]
        }
    },

    'ridge Regression' : {
        'model' : Ridge(),
        'parameters' : {
            'alpha' : np.arange(1,10)
        }
    },

    'lasso' : {
        'model' : Lasso(),
        'parameters': {
            'alpha':np.arange(1,10)
        }
    },

    'decision_tree' : {
        'model' : DecisionTreeRegressor(),
        'parameters': {
            'max_depth' : np.arange(1,10),
            'min_samples_split' : np.arange(1,10)
        }
    },

    'random forest' : {
        'model' : RandomForestRegressor(),
        'parameters': {
            'max_depth' : np.arange(1,10),
            'min_samples_split' : np.arange(1,10)
        }
    },
    'svr' : {
        'model' : SVR(),
        'parameters' : {
            'kernel': ['linear', 'rbf'],
            'C' : [1, 2, 3]
        }
    }
}

scores = []

for model_name,model in model_params.items():

    search = GridSearchCV(model['model'], model['parameters'],verbose=2, cv= 5)
    search.fit(df_x, df_y)
    scores.append({
        'model_name': model_name,
        'best_score' : search.best_score_,
        'best_parameters' : search.best_params_
    })

    for i in scores:
        print(i)

scores = []

search = GridSearchCV(SVR(),[{'kernel':['linear','rbf'],'C':[1,2,3]}], verbose= 2, cv = 5)
search.fit(df_x, df_y)

print(search.best_score_)
print(search.best_params_)

pipeline_linear_Regression = Pipeline([("Scaling",StandardScaler()),
                                        ("Linear Regression", LinearRegression(fit_intercept= True))])

pipeline_ridge_Regression = Pipeline([("Scaling",StandardScaler()),
                                    ("ridge_Regression",Ridge(alpha= 1))])

pipeline_lasso_Regression = Pipeline([("Scaling",StandardScaler()),
                                        ("lasso_Regression",Lasso(alpha= 1))])

pipeline_decisionTree = Pipeline([("Scaling",StandardScaler()),
                                    ("decisionTree",DecisionTreeRegressor(max_depth = 9, min_samples_split= 2))])

pipeline_randomForest = Pipeline([("Scaling",StandardScaler()),
                                      ("Random Forest",RandomForestRegressor(max_depth= 9, min_samples_split= 2))])

pipeline_SVR = Pipeline([("Scaling",StandardScaler()),
                        ("SVR",SVR(C=4000))])


pipelines = [pipeline_linear_Regression, pipeline_ridge_Regression, pipeline_lasso_Regression, pipeline_decisionTree,
             pipeline_randomForest, pipeline_SVR]


model_Dict = {0:"pipeline_linear_Regression", 1:"pipeline_ridge_Regression", 2: "pipeline_lasso_Regression", 3: "pipeline_decisionTree", 4: "randomForest", 5:"SVR"}

best = 0.0
index = 0

for pipe in pipelines:
    pipe.fit(x_train,y_train)

for i,model in enumerate(pipelines):
    print("{} gives the accuracy of {}".format(model_Dict[i],model.score(x_test, y_test)))

for i, model in enumerate(pipelines):
    y = model.score(x_test, y_test)
    if y > best:
        best = y
        index = i

print("Model with best performance is {} with accuracy score of {}".format(model_Dict[index],best))