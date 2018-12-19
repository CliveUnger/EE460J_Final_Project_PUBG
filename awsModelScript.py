import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pprint

# Read CSV files
train = pd.read_csv("train_V2.csv")
test = pd.read_csv("test_V2.csv")

train.drop(2744604, inplace=True)

trainId = train['Id']
testId  = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Concat training and set set for data exploration and feature engineering
y_train = train['winPlacePerc']

all_data = pd.concat((train.loc[:,'groupId':'winPoints'],
                     test.loc[:,'groupId':'winPoints']))

					 
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'					 
all_data["matchType"] = all_data['matchType'].apply(mapper)

# these features are inconsistent, lets just drop them.
# another possibility is to consolodate them into one column
all_data.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)

# Add a feature containing the number of players that joined each match.
all_data['playersJoined'] = all_data.groupby('matchId')['matchId'].transform('count')

# Number of players on a team (really the number of players that place together)
all_data = all_data.assign(players_in_team=all_data.groupby('groupId').groupId.transform('count'))

team_mapper = lambda x: 1 if ('solo' in x) else 2 if ('duo' in x) else 4
all_data["max_team_size"] = all_data['matchType'].apply(team_mapper)

print("Percentage of data that has more players in group then max_team_size:")
print(all_data[all_data["players_in_team"] > all_data["max_team_size"]].shape[0]/all_data.shape[0])

# Percentage of team that is filled
def team_fill(row):
    if row["players_in_team"] > row["max_team_size"]:
        return 1
    else:
        return row["players_in_team"]/row["max_team_size"]

all_data["team_fill_percentage"] = all_data.apply(team_fill, axis=1)

all_data.drop(['max_team_size'], axis=1, inplace=True)

# Heals + Boosts
all_data['healsAndBoosts'] = all_data['heals'] + all_data['boosts']

# Total Items Acquired
all_data['total_items_acquired'] = all_data['boosts'] + all_data['heals'] + all_data['weaponsAcquired']

# Headshot kill rate
all_data['headshot_kill_rate'] = all_data.headshotKills/all_data.kills

# KillPlace Percentage
all_data['killPlacePerc'] = all_data['killPlace']/all_data['maxPlace']

# teamwork
all_data['teamwork'] = all_data['assists'] + all_data['revives']

#Total amount of distance traveled
all_data['totalDistance'] = all_data['walkDistance'] + all_data['rideDistance'] + all_data['swimDistance']

# Number of kills without moving (possible cheater)
all_data['killsWithoutMoving'] = ((all_data['kills'] > 0) & (all_data['totalDistance'] == 0))

# Boosts Per Walk Distance
all_data['boostsPerWalkDistance'] = all_data['boosts']/(all_data['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.

# Heals Per Walk Distance
all_data['healsPerWalkDistance'] = all_data['heals']/(all_data['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.

# Heals and Boost Per Walk Distance
all_data['healsAndBoostsPerWalkDistance'] = all_data['healsAndBoosts']/(all_data['walkDistance']+1) #The +1 is to avoid infinity.

# Kills Per Walk Distance
all_data['killsPerWalkDistance'] = all_data['kills']/(all_data['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.

# Weapons Per Walk Distance
all_data["weaponsAcquiredPerDistance"] = all_data['weaponsAcquired']/(all_data['walkDistance']+1)

all_data['total_damage_by_team'] = all_data.groupby('groupId').damageDealt.transform('sum')
all_data['total_kills_by_team'] =  all_data.groupby('groupId').kills.transform('sum')
all_data['total_team_buffs'] = all_data.groupby('groupId').healsAndBoosts.transform('sum')

all_data['pct_killed'] = all_data.kills/(all_data.playersJoined - all_data.players_in_team + 1)
all_data['pct_knocked'] = all_data.DBNOs/(all_data.playersJoined - all_data.players_in_team + 1)
all_data['pct_kills_by_team'] = all_data.total_kills_by_team/(all_data.playersJoined - all_data.players_in_team + 1)

distance = all_data['totalDistance']

all_data['afk'] = ((distance == 0) & (all_data['kills'] == 0) 
                      & (all_data['weaponsAcquired'] == 0) 
                      & (all_data['matchType'] == 'solo')).astype(int)

all_data['cheater'] = ((all_data['kills'] / distance >= 1) 
                       | (all_data['kills'] > 30) 
                       | (all_data['roadKills'] > 10)).astype(int)
del distance
                      
pd.concat([all_data['afk'].value_counts(), all_data['cheater'].value_counts()], axis=1).T

# Create normalized features
all_data['killsNorm'] = all_data['kills']*((100-all_data['playersJoined'])/100 + 1)
all_data['damageDealtNorm'] = all_data['damageDealt']*((100-all_data['playersJoined'])/100 + 1)
all_data['maxPlaceNorm'] = all_data['maxPlace']*((100-all_data['playersJoined'])/100 + 1)

# Features to remove if you normalize
all_data = all_data.drop([ 'kills', 'damageDealt', 'maxPlace'],axis=1)

# Features to remove if you normalize
all_data = all_data.drop([ 'matchId', 'groupId', 'matchType'],axis=1)

# Fill empty values with 0
all_data.fillna(0, inplace=True)


# split back into train and test set
X_train = all_data[:trainId.size]
X_test = all_data[trainId.size:]
y_train = y_train

# Cross-Validation Abosolute Mean Error scoring
def abs_mean_error_cv(model, train_set):
    kf = KFold(5, shuffle=True, random_state=0).get_n_splits(train_set)
    abs_mean_err = cross_val_score(model, train_set, y_train, scoring="neg_mean_absolute_error", cv = kf)
    return abs_mean_err.mean()
	
	# Generates predictions and submission for a given model
def gen_sub(name, model):
    y_pred = model.predict(X_test)
    solution = pd.DataFrame({"Id":testId, "winPlacePerc":y_pred})
    solution.to_csv(name +".csv", index = False)


# Train basic model
'''m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)
m1.fit(X_train, y_train)'''

xgbregressor = xgb.XGBRegressor(learning_rate=0.11,
                                     max_depth = 5,
                                     subsample = 0.85,
                                     n_jobs=-1,
                                     colsample_bytree = 0.7,
                                     n_estimators=2000)

xgbregressor.fit(X_train, y_train)

gen_sub("xbboooooost", xgbregressor)






