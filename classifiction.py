# Load required packages for our analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans

target_var="y"
var_imp_cutoff=0.05
# STEP:1
path=file location
# Loard raw data into Python workspace.
master_data=pd.read_csv(path)


# STEP:2
# Remove unnecessary variables from the raw data.
master_data=master_data.drop('contact',axis=1)

# STEP:3
# Replace missing values.

# Replacing NA value sin categorical vaiables with mode.
nms=['job','marital','month','education','default','housing','loan','poutcome']

def rep_miss_cat(x,y):
    for i in list(range(len(y))):
        p=x[y[i]].value_counts()
        d=dict(p)
        x[y[i]]=x[y[i]].replace(np.NaN,list(d.keys())[0])
    return(x)

master_data[nms]=rep_miss_cat(master_data[nms],nms)

# Replacing NA value sin continuous vaiables with mean.
nms=['balance','duration','campaign','pdays','previous']
def rep_miss_cnt(x,y):
    for i in list(range(len(y))):
        p=x[y[i]].mean()
        x[y[i]]=x[y[i]].replace(np.NaN,p)
    return(x)

master_data[nms]=rep_miss_cnt(master_data[nms],nms)

# Replacing NA values in discreate vaiables with median.
nms=['age','day']
def rep_miss_dis(x,y):
    for i in list(range(len(y))):
        p=x[y[i]].median()
        x[y[i]]=x[y[i]].replace(np.NaN,p)
    return(x)

master_data[nms]=rep_miss_dis(master_data[nms],nms)

# STEP:4
# Cappping outliers.
# This one is not required for this analysis.

# Step:5
# Get summary statistics for all the continuous variables.
smry=master_data.describe()
smry.to_csv(path)


# STEP:5
# Perform label encoding.
le=LabelEncoder()
pp_master_data=master_data.apply(le.fit_transform)


# STEP:6
# Standadize the continuous variables.
scale = StandardScaler()
nms = ['balance','duration','campaign','pdays','previous']
pp_master_data[nms] = scale.fit_transform(pp_master_data[nms])


# STEP:7
# Split data into test and train.
train, test = train_test_split(pp_master_data, test_size=0.2)


# STEP:8
# Perform featureselection.

# Separating target variable from independent variables.
train_tv=train.loc[:,target_var]
train_iv=train.loc[:,train.columns.difference([target_var])]

#1. Remove constant variables from the data.
cons=train_iv.columns[train_iv.std()==0]
if(len(cons)>0):
    train_iv=train_iv.loc[:,train_iv.columns.difference([cons])]

#2. Considering only important variables into the model.
model = ExtraTreesClassifier()
model.fit(train_iv, train_tv)

imp_vars=list(train_iv.columns[model.feature_importances_>var_imp_cutoff])

# Final data with high important variables.
train_iv=train_iv.loc[:,imp_vars]

# Separatin indvars and target variable in test data.
test_iv=test[imp_vars]
test_tv=test[target_var]

# STEP:9
# Model Building.
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42,max_depth = 5)

# Train the model on training data
out=rf.fit(train_iv, train_tv)

# Predicting on test data.
y_pred=out.predict(test_iv)
y_pred=np.where(y_pred>=0.5,1,0)
print("Accuracy:",metrics.accuracy_score(test_tv, y_pred))
Accuracy: 0.9015813336282207

#===================================================
# Fitting gbm.

params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1}
clf = GradientBoostingClassifier(**params).fit(train_iv, train_tv)
gbm_y_pred=clf.predict(test_iv)
gbm_y_pred=np.where(gbm_y_pred>=0.5,1,0)
print("Accuracy:",metrics.accuracy_score(test_tv, gbm_y_pred))
Accuracy: 0.902134247484242
#===========================================================================
# Modeling with clusters approach
#===========================================================================
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(train_iv)
# Predicting the clusters on train data.
train_labels = kmeans.predict(train_iv)
# Predicting the clusters on test data.
test_labels = kmeans.predict(test_iv)

# Getting the cluster centers
C = kmeans.cluster_centers_
#========================================
# Train independent variables.
train_iv_1=train_iv.loc[train_labels==0,:]
train_iv_2=train_iv.loc[train_labels==1,:]
train_iv_3=train_iv.loc[train_labels==2,:]
train_iv_4=train_iv.loc[train_labels==3,:]
# Train dependent variables.
train_d_1=train_tv.loc[train_labels==0]
train_d_2=train_tv.loc[train_labels==1]
train_d_3=train_tv.loc[train_labels==2]
train_d_4=train_tv.loc[train_labels==3]
#========================================
# Test independent variables.
test_1=test_iv.loc[test_labels==0,:]
test_2=test_iv.loc[test_labels==1,:]
test_3=test_iv.loc[test_labels==2,:]
test_4=test_iv.loc[test_labels==3,:]
#========================================
# Test dependent variables.
test_1_d=test_tv.loc[test_labels==0]
test_2_d=test_tv.loc[test_labels==1]
test_3_d=test_tv.loc[test_labels==2]
test_4_d=test_tv.loc[test_labels==3]
#========================================

# Randomforest and gbm on train data sets.
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42,max_depth = 5)

rf_tr_out_1=rf.fit(train_iv_1, train_d_1)
rf_tr_out_2=rf.fit(train_iv_2, train_d_2)
rf_tr_out_3=rf.fit(train_iv_3, train_d_3)
rf_tr_out_4=rf.fit(train_iv_4, train_d_4)

# Predicting models on test data sets.
rf_ts_out_1=rf_tr_out_1.predict(test_1)
rf_ts_out_2=rf_tr_out_2.predict(test_2)
rf_ts_out_3=rf_tr_out_3.predict(test_3)
rf_ts_out_4=rf_tr_out_4.predict(test_4)

# Getting overall performance of rf model.
rf_test_out=np.concatenate([rf_ts_out_1,rf_ts_out_2,rf_ts_out_3,rf_ts_out_4])
rf_test_tv=np.concatenate([test_1_d,test_2_d,test_3_d,test_4_d])
print("Accuracy:",metrics.accuracy_score(rf_test_tv, rf_test_out))
Accuracy: 0.8895278115669578

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# gbm on train data sets.
params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1}
clf = GradientBoostingClassifier(**params).fit(train_iv, train_tv)

gb_tr_out_1=clf.fit(train_iv_1, train_d_1)
gb_tr_out_2=clf.fit(train_iv_2, train_d_2)
gb_tr_out_3=clf.fit(train_iv_3, train_d_3)
gb_tr_out_4=clf.fit(train_iv_4, train_d_4)

# Predicting models on test data sets.
gb_ts_out_1=gb_tr_out_1.predict(test_1)
gb_ts_out_2=gb_tr_out_2.predict(test_2)
gb_ts_out_3=gb_tr_out_3.predict(test_3)
gb_ts_out_4=gb_tr_out_4.predict(test_4)

# Getting overall performance of gbm model.
gb_test_out=np.concatenate([gb_ts_out_1,gb_ts_out_2,gb_ts_out_3,gb_ts_out_4])
print("Accuracy:",metrics.accuracy_score(rf_test_tv, gb_test_out))
Accuracy: 0.8863209112020347

