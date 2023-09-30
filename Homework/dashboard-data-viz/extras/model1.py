import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('../data/baseball.csv')

# If we want to create a model for predictions

#model = LogisticRegression()

#model.fit(X, y)


#pickle.dump(model, open('../models/model.pkl', 'wb') )