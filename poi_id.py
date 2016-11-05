#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.preprocessing import StandardScaler, MinMaxScaler    
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

    ##############################
    ### Task 1: Select features### 
    ##############################

print ""    
print "# TASK 1: Select which features will be used. \n"

poi = "poi"

features_email = ["from_messages",
    "from_poi_to_this_person",
    "from_this_person_to_poi",
    "shared_receipt_with_poi",
    "to_messages"]

features_financial = ["bonus",
    "deferral_payments",
    "deferred_income",
    "director_fees",
    "exercised_stock_options",
    "expenses",
    "loan_advances",
    "long_term_incentive",
    "other",
    "restricted_stock",
    "restricted_stock_deferred",
    "salary",
    "total_payments",
    "total_stock_value"]

features_list = [poi] + features_email + features_financial


### Dataset Exploration
def count():
    No_employees = 0
    No_POI = 0
    No_NonPOI=0
    for i in data_dict:
        No_employees = No_employees + 1
        if data_dict[i]["poi"] == True:
            No_POI = No_POI + 1
    No_NonPOI = No_employees - No_POI
    return No_employees, No_POI, No_NonPOI
print ""
print "Number of employees, POI's and NON_POI's in the dataset: ", count()
print ""


print "List of Features & NaNs"
No_NaaNs = {}
for i in features_list:
    No_NaaNs[i] = 0
    for k in data_dict.keys():
        if data_dict[k][i] == "NaN":
            No_NaaNs[i] += 1
    print(i, No_NaaNs[i])
print ""

       
    ###############################
    ### Task 2: Remove outliers ###
    ###############################

    
print "# TASK 2: Remove outliers. \n"

### Bonus vs salary plotting looking for possible outliers 

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for i in data:
    salary = i[0]
    bonus = i[1]
    plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")

### Uncomment next line to show plot
#plt.show()
## plot shows TOTAL is an outlier

### Searching for more outliers by employee names inspection
employee_names = []
for i in data_dict:
    employee_names.append(i)

employee_names.sort()

### Uncomment next 3 lines to print Employees names
#print"Employees names :"
#pprint.pprint(employee_names)
#print ""
# Travel Agency seems irrelevant. All names except for Total and Travel Agency seem valid

### "TOTAL" and "THE TRAVEL AGENCY IN THE PARK: " inspection
print "TOTAL: "
pprint.pprint(data_dict["TOTAL"])
print""
print "THE TRAVEL AGENCY IN THE PARK: "
pprint.pprint(data_dict["THE TRAVEL AGENCY IN THE PARK"])
print""

### Checking for employees whose all features are NaN's 
All_NaN = []
for i in data_dict:
    num = 0
    for k in features_list:
        if data_dict[i][k] == "NaN":
            num += 1
    if num == len(features_list) - 1:
        All_NaN.append(i)
print "Employees names whose all features are NaN:", All_NaN
print ""


### Removing the 3 outliers from the dataset
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E")
print "Number of employees Updated: ",len(data_dict)
print ""
    

    #####################################
    ### Task 3: Create new feature(s) ###
    #####################################

    
print "# TASK 3: Create new features \n"
  
### Store to my_dataset for easy export below.
my_dataset = data_dict    
    
### Create 2 new features, fraction_from_POI & fraction_to_POI and insert them into data_dict

for i,i in data_dict.items():
    if i["from_poi_to_this_person"] != "NaN" or i["from_this_person_to_poi"] != "NaN":
        i["fraction_from_POI"] = i["from_poi_to_this_person"] / float(i["to_messages"])
        i["fraction_to_POI"] = i["from_this_person_to_poi"] / float(i["from_messages"])
    else:
        i["fraction_from_POI"] = 0
        i["fraction_to_POI"] = 0

features_list.append("fraction_from_POI")
features_list.append("fraction_to_POI")

### Select k-best features
def my_k_best(data_dict,features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores) 
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    print "k_best.scores: \n", sorted_pairs
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features.keys()

best_features = my_k_best(my_dataset,features_list, 3)
best_features.insert(0, "poi")
print ""
print "Best {0} selected features: {1}\n".format(len(best_features) , best_features[0:])

### Update the features list withe best k features
features_list=best_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
### Split into labels and features
labels, features = targetFeatureSplit(data)

### Split train and test data
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


    ###########################################
    ### Task 4: Try a varity of classifiers ###
    ###########################################

    
print "# TASK 4: Try a varity of classifiers "

### Gaussian naive bayes
G_clf = GaussianNB()
G_clf.fit(features_train,labels_train)
predG = G_clf.predict(features_test)
print "\n Gaussian naive bayes scores "
print "accuracy = ", accuracy_score(labels_test,predG)
print "precision = ", precision_score(labels_test,predG)
print "recall = ", recall_score(labels_test,predG)
#test_classifier(G_clf, my_dataset, features_list)

### Decision Trees
DT_clf = DecisionTreeClassifier()
DT_clf.fit(features_train,labels_train)
predDT = DT_clf.predict(features_test)
print "\n Decision Trees scores "
print "accuracy = ", accuracy_score(labels_test,predDT)
print "precision = ", precision_score(labels_test,predDT)
print "recall = ", recall_score(labels_test,predDT)
test_classifier(DT_clf, my_dataset, features_list)

### K nearest neighbors
KNN_clf = KNeighborsClassifier()
KNN_clf.fit(features_train,labels_train)
predKN = KNN_clf.predict(features_test)
print "\n KNeighbors scores "
print "accuracy = ", accuracy_score(labels_test,predKN)
print "precision = ", precision_score(labels_test,predKN)
print "recall = ", recall_score(labels_test,predKN)
print ""
#test_classifier(KNN_clf, my_dataset, features_list)


    #####################################
    ### Task 5: Tune your classifiers ###
    #####################################
 
print "# TASK 5: Tune your classifiers \n"

### Tune Decision Tree Classifier and get the best estimator to apply on the tester
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

#parameters = {}
parameters = {"splitter": ["best"],"criterion": ["gini"], "min_samples_leaf": [1,2],
"max_depth": [None],
"min_samples_split": [1,2]}
## Those are the best.params, entered after running the script once

a = DecisionTreeClassifier()
### Creating the Gridsearch 
DT_Tuned_clf = grid_search.GridSearchCV(a, parameters, cv = cv, scoring = "f1")
DT_Tuned_clf.fit(features, labels)
### Printing out the best parameters
print "DT_Tuned_clf.best_params_"
print DT_Tuned_clf.best_params_
print ""

clf = DT_Tuned_clf.best_estimator_
test_classifier(clf, my_dataset, features_list)


    #############################
    ### Task 6: Results check ###
    #############################

    
dump_classifier_and_data(clf, my_dataset, features_list)


print "End of Report"