#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#sklearn.metrics.ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, precision_recall_curve, auc,ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:


pip install --upgrade scikit-learn


# In[3]:


data = "healthcare-dataset-stroke-data (1).csv"


stroke = pd.read_csv(data, sep = ',')
                  
stroke.head()




# In[4]:


stroke.columns


# In[5]:


#stroke = stroke.drop(columns = ['id'])
stroke


# In[6]:


#stroke.isnull()
stroke=stroke.fillna(stroke['bmi'].mean())
stroke.head()


# In[7]:


stroke.describe()


# In[8]:


stroke.isnull().sum().to_frame(name="missing").sort_values(by="missing", ascending=False).style.background_gradient(cmap='Reds')


# In[9]:


#Data description

plt.rc("font", size = 15)
stroke.gender.value_counts(sort = True).plot(kind = 'bar')
plt.title ('Gender Distribution of Stroke Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.plot()
plt.savefig('Rating Distribution.jpg', dpi = 100)


# In[10]:


#rating distribution
# visualizing ratings from zero to 10
plt.rc("font", size = 15)
stroke.ever_married.value_counts(sort = False).plot(kind = 'bar', width = 0.5, color = 'red')
plt.title (' Distribution by Marital Status')
plt.xlabel('Marital status')
plt.ylabel('Count')
plt.plot()
plt.savefig('dismar.png', dpi = 100)


# In[11]:


#rating distribution
# visualizing ratings from zero to 10
plt.rc("font", size = 15)
stroke.stroke.value_counts(sort = True).plot(kind = 'bar', width = 0.5, color = 'green')
plt.title (' Distribution of data by previous experience of stroke ')
plt.xlabel('Previous Incidence of stroke ')
plt.ylabel('Count')
plt.plot()
plt.savefig('dismar.png', bbox_inches='tight', dpi = 100)


# In[12]:


#rating distribution
# visualizing ratings from zero to 10
plt.rc("font", size = 15)
stroke.work_type.value_counts(sort = True).plot(kind = 'bar', width = 0.5, color = 'purple')
plt.title (' Distribution of data by type of work ')
plt.xlabel('Work type')
plt.ylabel('Count')
plt.plot()
plt.savefig('dismar.png',bbox_inches='tight', dpi = 100)


# In[13]:


fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6) , squeeze=True)

sns.boxplot(data=stroke,y=stroke['age'],x=stroke['stroke'],palette='tab10' , ax=axes[0])
sns.boxplot(data=stroke,y=stroke['bmi'],x=stroke['stroke'],palette='tab10' , ax=axes[1])
sns.boxplot(data=stroke,y=stroke['avg_glucose_level'],x=stroke['stroke'],palette='tab10' , ax=axes[2])

plt.show


# In[ ]:





# In[14]:


first = stroke["stroke"]
first


# In[15]:


print(stroke['gender'].unique())
stroke.gender.value_counts()






# In[16]:


def age_cohort(age):
    if   age >= 0 and age <= 20:
        return "0-20"
    elif age > 20 and age <= 40:
        return "20-40"
    elif age > 40 and age <= 50:
        return "40-50"
    elif age > 50 and age <= 60:
        return "50-60"
    elif age > 60:
        return "60+"


# In[17]:


df_copy = stroke.copy()
df_copy['age_group'] = df_copy['age'].apply(age_cohort)
df_copy.sort_values('age_group', inplace = True)


# In[18]:


plt.figure(figsize=(12,8))
df_copy.age_group.value_counts().plot.pie(autopct="%.1f%%", wedgeprops={"linewidth":2,"edgecolor":"white"});
plt.title("Distribution by age")
plt.show()


# In[19]:


def plot_pie_value_count(target_column, df, label_dict = None, autopct = '%1.1f%%', title = 'Chart', title_fontweight = 'bold', 
                         title_fontstyle = 'italic', title_fontsize  = 14):
    labels = []
    if isinstance(label_dict, dict):
        for i in range(df[target_column].value_counts().shape[0]):
            labels.append(label_dict[df[target_column].value_counts().index[i]])
    elif label_dict is None:
        labels = df[target_column].value_counts().index
    else:
        raise NotImplementedError('Pass a Dictionary or None')
    plt.pie(df[target_column].value_counts(), labels = labels, autopct = autopct)
    plt.title(title, fontsize = title_fontsize, fontweight = title_fontweight, fontstyle = title_fontstyle)
    plt.show()


# In[20]:


marriage_labels = {
    'Yes' : 'Married atleast once',
    'No'  : 'Never Married',
}
plot_pie_value_count('ever_married', stroke[stroke['age'] > 18], label_dict = marriage_labels, title = 'Distribution by Marital status')


# In[21]:


plot_pie_value_count('gender', stroke, title = 'Gender Representation')


# In[22]:


plot_pie_value_count('Residence_type', stroke, title = 'Urban-Rural Distribution')


# In[23]:


df = stroke


# In[24]:


def plot_chances(
    target_column,
    chance_column,
    df,
    labels = None,
    x_tick_labels = None,
    figsize  = (12, 8),
    x_label = 'X',
    y_label = 'Y',
    title = 'Title',
    title_fontweight = 'bold',
    title_fontstyle = 'italic',
    title_fontsize = 15,
    ):
    '''
        Input:- target_column    : (string type) Name of column whose events are to be plotted on x-axis
                chance_column    : (string type) Name of column which denotes occurance of single event using 0 and 1. Conditional 
                                   probability of this column is to plotted.
                df               : (Pandas DataFrame type) Dataframe which contains target_column and chance_column as columns
                labels           : (tuple or None type) This contain the arguments for range object. This is provided to give labels
                                   in case of continuous numerical data in target_column.
                x_tick_labels    : (dict or None type) It is a dictionary mapping unique values of target_column with custom labels we
                                   want as x-tick labels
                figsize          : (tuple of int) It contains the size of figure we want for the plot
                x_label          : (string type) Denotes label for x-axis
                y_label          : (string type) Denotes label for y-axis
                title            : (string type) Denotes the title of plot
                title_fontweight : (string type) Used to set the fontweight argument of plt.title()
                title_fontstyle  : (string type) Used to set the fontstyle argument of plt.title()
                title_fontsize   : (string type) Used to set the fontsize argument of plt.title()

        Computes the coonditional probability of occurance of event represented in chance_column, given that a unique event from 
        target_column took place.

        Displays matplotlib bar plots of the computed probability. The probability is plotted along y-axis while the events from 
        target_column are along x_axis.
    '''
    if labels is None:
        labels = list(df[target_column].unique())
    if x_tick_labels is None:
        x_tick_labels = labels
    fig, ax = plt.subplots(figsize = figsize, )
    if isinstance(labels, tuple):
        start = labels[0]
        end = labels[1]
        step = labels[2]
        start_labels = np.arange(start, end, step)
        for i in np.arange(len(start_labels)):
            plt.bar(i, df[(df[target_column] > start_labels[i]) & (df[target_column] < (start_labels[i] + step))][chance_column].mean())
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(np.arange(len(start_labels)))
        x_tick_labels = [f'{i}-{i+step}' for i in start_labels]
        ax.set_xticklabels(x_tick_labels)
    else:
        for i in np.arange(len(labels)):
            plt.bar(i, df[df[target_column] == labels[i]][chance_column].mean())
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(np.arange(len(labels)))
        if isinstance(x_tick_labels, dict):
            x_tick_labels_list = [i for i in np.arange(len(labels))]
            for i in labels:
                x_tick_labels_list[labels.index(i)] = x_tick_labels[i]
            ax.set_xticklabels(x_tick_labels_list)
        else:
            ax.set_xticklabels(x_tick_labels)

    plt.title(title, fontweight = title_fontweight, fontstyle = title_fontstyle, fontsize = title_fontsize)
    plt.show()


# In[25]:


plot_chances('gender',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_label = 'Gender',
y_label = 'Chance of stroke',
title = 'Gender vs Chance of Stroke',
)


# In[26]:


yesno = {
    1 : 'Yes',
    0 : 'No'
}
plot_chances('hypertension',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_tick_labels = yesno,
x_label = 'Hypertension',
y_label = 'Chance of stroke',
title = 'Hypertension vs Chances of Stroke',
)


# In[27]:


yesno = {
    1 : 'Yes',
    0 : 'No'
}
plot_chances('heart_disease',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_tick_labels = yesno,
x_label = 'Heartdisease',
y_label = 'Chance of stroke',
title = 'Heartdisease vs Chance of Stroke',
)


# In[28]:


plot_chances('ever_married',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_label = 'Ever Married',
y_label = 'Chance of stroke',
title = 'Marriage and Chance of Stroke',
)


# In[29]:


x_ticks = np.array([i for i in range(5, 105, 10)])
age = [(i, i+10) for i in range(0, 100, 10)]
plt.bar(x_ticks - 1, 
        [df[(df['age'] > age[i][0]) & (df['age'] < age[i][1]) & (df['ever_married'] == 'No')]['stroke'].mean() for i in range(10)],
        width=2, label = 'No')
plt.bar(x_ticks + 1,
        [df[(df['age'] > age[i][0]) & (df['age'] < age[i][1]) & (df['ever_married'] == 'Yes')]['stroke'].mean() for i in range(10)],
        width=2, label = 'Yes')

age_labels = [f'{age[i][0]}-{age[i][1]}' for i in range(len(age))]
plt.xticks(x_ticks, labels = age_labels)
plt.xlabel('Age groups')
plt.ylabel('Chance of Stroke')
plt.legend(title = 'Ever Married')
plt.title('Marriage vs chance of stroke for different age groups')

plt.show()


# In[30]:


plot_chances('work_type',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_label = 'Work type',
y_label = 'Chance of stroke',
title = 'Work type vs Chance of Stroke',
)


# In[31]:


plot_chances('Residence_type',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_label = 'Residence type',
y_label = 'Chance of stroke',
title = 'Residence type vs Chance of Stroke',
)


# In[32]:


plot_chances('smoking_status',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
x_label = 'Smoking Status',
y_label = 'Chance of stroke',
title = 'Smoking Status vs Chance of Stroke',
)


# In[33]:


plot_chances('age',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
labels = (0, 100, 10),
x_label = 'Age (in years)',
y_label = 'Chance of stroke',
title = 'Age vs Chance of Stroke',
)


# In[34]:


plot_chances('bmi',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
labels = (15, 60, 5),
x_label = 'BMI',
y_label = 'Chance of stroke',
title = 'BMI vs Chance of Stroke',
)


# In[35]:


plot_chances('avg_glucose_level',
chance_column = 'stroke',
df = df,
figsize = (8, 5),
labels = (30, 330, 30),
x_label = 'Average glucose level (in mg/dl)',
y_label = 'Chance of stroke',
title = 'Average glucose level vs Chance of Stroke',
)


# In[36]:


plot_chances('avg_glucose_level',
chance_column = 'stroke',
df = df[df['avg_glucose_level'] < 270],
figsize = (8, 5),
labels = (30, 270, 30),
x_label = 'Average glucose level (in mg/dl)',
y_label = 'Chance of stroke',
title = 'Average glucose level vs Chance of Stroke',
)


# In[37]:


stroke.isnull().mean()
df.drop('id', axis=1, inplace=True)
df


# In[38]:


#Categorical Encoding is a process where we transform categorical data into numerical data.
#convert in binary columns with 2 results
from sklearn import preprocessing

columns_obj = ["gender", "ever_married" ,"Residence_type"]
encoding = preprocessing.LabelEncoder()
for col in columns_obj:
    df[col]=  encoding.fit_transform(df[col])
#convert in 0 and 1 the rest of columns    
df = pd.get_dummies(df)
df.head()


# In[39]:


import os
cwd = os.getcwd()
cwd


# In[40]:


pip install -U imbalanced-learn


# In[41]:


conda install -c conda-forge imbalanced-learn


# In[42]:


pip install Tensorflow


# In[43]:


pip install imblearn 


# In[44]:


from imblearn import under_sampling, over_sampling

#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE

import os, colorama
from colorama import Fore,Style,Back #specifying all 3 types
os.system("mode con: cols=120 lines=30")


# In[45]:


print 


# In[46]:


#change outliers value to approximate maximum
#X_train.loc[X_train.bmi >= 53.4, 'bmi'] = 49


# In[47]:


#sepate labels and target
X = df.drop(columns = ['stroke'])
#target
y = df['stroke']
#oversample data
smote = SMOTE(random_state=42)
X , y = smote.fit_resample(X,y)

before = df.stroke.value_counts(normalize=True)
after = y.value_counts(normalize=True)
print(Fore.BLACK + 'Rows before smote:' + Fore.GREEN + ' {}'.format(df.shape[0]))
print(Fore.BLACK + 'Rows after smote:' + Fore.GREEN + ' {}'.format(X.shape[0]))


# In[48]:


df.head()


# In[49]:


df.dtypes


# In[50]:


# let's separate into training and testing set
X_train, X2, y_train, y2 = train_test_split(
    X,  # predictors
    y,  # target
    test_size=0.30, #size of test data
    shuffle=True, #shuffe rows
    stratify=y,# makes a split so that the proportion of values in the sample
    random_state=42)  # seed to ensure reproducibility

X_val, X_test, y_val, y_test = train_test_split(
    X2, y2, test_size=0.50, shuffle=True, stratify=y2, random_state=42)
#check rows and columns
print(Fore.BLACK + 'Train set shape:' + Fore.GREEN + ' {}'.format(X_train.shape))
print(Fore.BLACK + 'Validation set shape:' + Fore.GREEN + ' {}'.format(X_val.shape))
print(Fore.BLACK + 'Test set shape:' + Fore.GREEN + ' {}'.format(X_test.shape))


# In[51]:


pip install plotly


# In[52]:


from plotly.subplots import make_subplots
import plotly.graph_objs as go


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[54]:


# Set up the subplots grid
fig = make_subplots(rows=1, cols=3, 
                    # Set the subplot titles
                    subplot_titles=['Age', 'Avg glucose level', 'BMI'])
#create boxplot visualization of numeric columns
fig.add_trace(go.Box(x=X_train.age, name='', showlegend=False), row=1, col=1)
fig.add_trace(go.Box(x=X_train.avg_glucose_level, name='', showlegend=False), row=1, col=2)
fig.add_trace(go.Box(x=X_train.bmi, name='', showlegend=False), row=1, col=3)
#config size
fig.update_layout(height=400,font_family='Verdana',paper_bgcolor='#edeae7',plot_bgcolor='#edeae7')
#show visualizations
fig.show()


# In[55]:


conda install -c conda-forge feature_engine


# In[56]:


from feature_engine.selection import SelectBySingleFeaturePerformance


# In[ ]:





# In[57]:


# set up a machine learning model
rf = RandomForestClassifier(
    n_estimators=10, random_state=1, n_jobs=4)

# set up the selector
#it trains a machine learning model for every single feature
sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=rf,
    scoring="roc_auc",
    cv=3,
    threshold=0.5)

# find predictive features
sel.fit(X_train, y_train)

#performance of columns
sel.feature_performance_


# In[58]:


import plotly.express as px


# In[59]:


#plot feature importance
x1 = pd.Series(sel.feature_performance_).sort_values(ascending=False)
#create the plot
fig = px.bar(x=x1.values, color=x1.values, y=x1.index,color_continuous_scale='Teal')
fig.update_layout(title_x=0.5,title_text=(f"Feature performance"), height=400,width =600,font_family='Verdana',
                 font=dict(family="Verdana,Verdana",size=13),yaxis_title=None, xaxis_title='ROC AUC')
    
#config plot
fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False,
                 marker_line_color="black")
#config plot
fig.update_coloraxes(showscale=False)
fig.show()


# In[60]:


from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
kf = KFold(n_splits=5)
kf.get_n_splits(X)


# In[61]:


def cf_matrix_model(name, model):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title(name);
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()


# In[62]:


#logistic regression
lr_model = LogisticRegression(solver='liblinear',random_state=42, max_iter=1000)
#parameters
lr_param = {'penalty': ['l1', 'l2'],'C':[0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 15]}
#
grid_lr = GridSearchCV(lr_model, lr_param ,scoring = 'roc_auc', cv= 5,n_jobs=-1)
#gridsearch
search_lr = grid_lr.fit(X_train, y_train)
#best parameter
best_lr = search_lr.best_estimator_
#get score
cross_lr =  cross_val_score(
    best_lr,
    X_val, 
    y_val,
    n_jobs=-1,
    scoring='accuracy',
    cv=kf, # k-fold
)
#dataframe metrics
lr_accu = pd.DataFrame(data={'Score': cross_lr, 'Metric': 'Accuracy', 'Model': 'Logistic Regression'})

print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + 'Logistic Regression')
print(Back.RESET)
print(Fore.BLUE + 'Best AUC: ' + Fore.GREEN + str(round(grid_lr.best_score_,2)))
print(Fore.BLUE + 'Mean validation set accuracy: ' + Fore.GREEN + str(round(cross_lr.mean()*100, 2)) +"%")
print(Fore.BLUE + 'Standard deviation: ' + Fore.GREEN + str(round(cross_lr.std()*100, 2)))

cf_matrix_model("Logistic Regression", best_lr, )


# In[63]:


#fold configuration
kf = KFold(n_splits=5, shuffle=True, random_state=4)
#Random Forest
rf_model = RandomForestClassifier(random_state=42)
#parameters
rf_param = {'n_estimators':[50, 100, 200, 500, 1000],'max_depth':[3, 4, 5,8]}

#gridsearch
grid_rf = GridSearchCV(rf_model,rf_param, scoring='roc_auc', cv=kf,n_jobs=-1)
#fit
search_rf = grid_rf.fit(X_train, y_train)
#get best parameter
best_rf = search_rf.best_estimator_

#get score
cross_rf =  cross_val_score(
    best_rf,
    X_val, 
    y_val,
    scoring='accuracy',
    cv=kf, # k-fold
    n_jobs=-1
)
#dataframe metrics
rf_accu = pd.DataFrame(data={'Score': cross_rf, 'Metric': 'Accuracy', 'Model': 'Random Forest'})

print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + 'Random Forest')
print(Back.RESET)
print(Fore.BLUE + 'Best AUC: ' + Fore.GREEN + str(round(grid_rf.best_score_,2)))
print(Fore.BLUE + 'Mean validation set accuracy: ' + Fore.GREEN + str(round(cross_rf.mean()*100, 2)) +"%")
print(Fore.BLUE + 'Standard deviation: ' + Fore.GREEN + str(round(cross_rf.std()*100, 2)))

def cf_matrix_model(name, model):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title(name);
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()

cf_matrix_model("Random Forest", best_rf)


# In[64]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[65]:


#Decision tree
tree_model= DecisionTreeClassifier(random_state=42)
#parameters
tree_param = {'max_features': ['auto', 'sqrt', 'log2'],'ccp_alpha': [0.1, .01, .001, 1.0],
              'max_depth' : [5, 6, 7, 8, 9], 'criterion' :['gini', 'entropy']}
# gridsearch
grid_tree = GridSearchCV(tree_model, tree_param, scoring = 'roc_auc' ,cv=5,n_jobs=-1)
#fit gridsearch
search_tree = grid_tree.fit(X_train, y_train)
#best parameters
best_tree = search_tree.best_estimator_
#get score
cross_tree =  cross_val_score(
    best_tree,
    X_val, 
    y_val,
    n_jobs=-1,
    scoring='accuracy',
    cv=kf, # k-fold
)
#dataframe metrics
tree_accu = pd.DataFrame(data={'Score': cross_tree, 'Metric': 'Accuracy', 'Model': 'Decision Tree'})

print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + 'Decision Tree')
print(Back.RESET)
print(Fore.BLUE + 'Best AUC: ' + Fore.GREEN + str(round(grid_tree.best_score_,2)))
print(Fore.BLUE + 'Mean validation set accuracy: ' + Fore.GREEN + str(round(cross_tree.mean()*100, 2)) +"%")
print(Fore.BLUE + 'Standard deviation: ' + Fore.GREEN + str(round(cross_tree.std()*100, 2)))

cf_matrix_model("Decision Tree", best_tree)


# In[66]:


#concat metrics
metrics=pd.concat([rf_accu, lr_accu,tree_accu], axis=0)
metrics['Score']=metrics.Score.mul(100)
#plot configuration
fig = px.box(metrics, x="Model", y="Score", color="Metric", 
             title="Accuracy of models on Validation set",
             color_discrete_sequence = ['#d5a036'])
#plot configuration
fig.update_layout(title_x=0.5, xaxis_title='', yaxis_ticksuffix='%',font_family='Verdana',
                 font=dict(family="Verdana,Verdana",size=12),height=500, width=700)
#plot configuration
fig.update_xaxes(categoryorder='median descending')

#config opacity
fig.update_traces(opacity=0.80)

fig.show()


# In[67]:


# ROC Curves
fpr = {}
tpr = {}
roc_auc = {}
thresh = {}
#models
models=[best_rf, best_lr, best_tree]

#fill values
for i in range(len(models)):
    m=models[i]
    y_probs=m.predict_proba(X_test)
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_probs[:,1], pos_label=1)
    roc_auc[i] = cross_val_score(m, X_test, y_test, cv=kf, 
                                 scoring='roc_auc', n_jobs=-1).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr[0], y=tpr[0], line=dict(color='#B25068', width=2.5), opacity=0.7,
                         hovertemplate = 'Random Forest True positive rate = %{y:.3f}, False positive rate = %{x:.3f}<extra></extra>',
                         name='Random Forest  (AUC = {:.3f})'.format(roc_auc[0])))

fig.add_trace(go.Scatter(x=fpr[1], y=tpr[1], line=dict(color='#3AB0FF', width=2.5), opacity=0.7,
                         hovertemplate = 'Logistic Regression True positive rate = %{y:.3f}, False positive rate = %{x:.3f}<extra></extra>',
                         name='Logistic Regression  (AUC = {:.3f})'.format(roc_auc[1])))


fig.add_trace(go.Scatter(x=fpr[2], y=tpr[2], line=dict(color='#F87474', width=2.5), opacity=0.8,
                         hovertemplate = 'Decision Tree True positive rate = %{y:.3f}, False positive rate = %{x:.3f}<extra></extra>',
                         name='Decision Tree  (AUC = {:.3f})'.format(roc_auc[2])))

fig.add_shape(type="line", xref="x", yref="y", x0=0, y0=0, x1=1, y1=1, 
              line=dict(color="Black", width=1, dash="dot"))

fig.update_layout( title_x=0.5, title_text="Comparing ROC Curve on the Test Set", hovermode="x unified",font_family='Verdana',
                  xaxis_title='False Positive Rate', yaxis_title='True Positive Rate ', font=dict(family="Verdana,Verdana",size=12),
                  legend=dict(y=.12, x=1, xanchor="right",bordercolor="black",borderwidth=1, font=dict(size=12)),
                  height=500, width=700)


# In[68]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def Metrics(model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(Fore.BLUE +'Precision: '+ Fore.GREEN + str(precision_score(y_test, y_pred)))
    print(Fore.BLUE +'Recall: ' +Fore.GREEN + str(recall_score(y_test, y_pred)))
    print(Fore.BLUE +'F1: ' +Fore.GREEN + str(f1_score(y_test, y_pred)))
    print(Fore.BLUE +'Accuracy: ' + Fore.GREEN +  str(accuracy_score(y_test, y_pred)))
print(search_rf.best_estimator_)

Metrics(best_rf)

def Metrics(model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(Fore.BLUE +'Precision: '+ Fore.GREEN + str(precision_score(y_test, y_pred)))
    print(Fore.BLUE +'Recall: ' +Fore.GREEN + str(recall_score(y_test, y_pred)))
    print(Fore.BLUE +'F1: ' +Fore.GREEN + str(f1_score(y_test, y_pred)))
    print(Fore.BLUE +'Accuracy: ' + Fore.GREEN +  str(accuracy_score(y_test, y_pred)))
print(search_tree.best_estimator_)

Metrics(best_tree)


# In[69]:


def Metrics(model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(Fore.BLUE +'Precision: '+ Fore.GREEN + str(precision_score(y_test, y_pred)))
    print(Fore.BLUE +'Recall: ' +Fore.GREEN + str(recall_score(y_test, y_pred)))
    print(Fore.BLUE +'F1: ' +Fore.GREEN + str(f1_score(y_test, y_pred)))
    print(Fore.BLUE +'Accuracy: ' + Fore.GREEN +  str(accuracy_score(y_test, y_pred)))
print(search_lr.best_estimator_)

Metrics(best_lr)


# In[70]:


df.replace(to_replace="Urban", value=1, inplace=True)
df.replace(to_replace="Rural", value=0, inplace=True)

df.replace(to_replace="Yes", value=1, inplace=True)
df.replace(to_replace="No", value=0, inplace=True)


# In[71]:


df.replace(to_replace="Urban", value=1, inplace=True)
df.replace(to_replace="Rural", value=0, inplace=True)

df.replace(to_replace="Yes", value=1, inplace=True)
df.replace(to_replace="No", value=0, inplace=True)
plt.figure(figsize=(18, 9))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[72]:


plt.savefig("output.jpg")


# In[73]:


#They seem very uncorrelated
plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams.update({"font.size":8})
#There don't seem to be many correlations (big ones)
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette('coolwarm',as_cmap=True))


# In[74]:


from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


# In[75]:


#tree.plot_tree(clf)


# In[ ]:





# In[76]:


fig = px.scatter(df, x="avg_glucose_level", y="bmi", color="stroke", title="Stroke Sample Distribution Based on BMI ans AVG. Glucose Level")
fig.show()
fig = px.scatter(df, x="age", y="bmi", color="gender", facet_col="stroke", title="Stroke Distribution Based on BMI and Age")
fig.show()


# In[ ]:





# In[ ]:





# In[77]:


pip install keras 


# In[78]:


pip install python-sklearn 


# In[79]:


from tensorflow.keras import Sequential


# In[80]:


#from standard_plots import *
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
#import theano


# In[81]:


dataset = df
dataset.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




