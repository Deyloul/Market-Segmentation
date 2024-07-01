import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import warnings
warnings.filterwarnings("ignore")
import os

# Loading the dataset #
df=pd.read_csv('C:/Customer Data.csv')

# Checking The head and tail of the dataset #
df.head()
  CUST_ID      BALANCE  ...  PRC_FULL_PAYMENT  TENURE
0  C10001    40.900749  ...          0.000000      12
1  C10002  3202.467416  ...          0.222222      12
2  C10003  2495.148862  ...          0.000000      12
3  C10004  1666.670542  ...          0.000000      12
4  C10005   817.714335  ...          0.000000      12

[5 rows x 18 columns]
df.tail()
     CUST_ID     BALANCE  ...  PRC_FULL_PAYMENT  TENURE
8945  C19186   28.493517  ...              0.50       6
8946  C19187   19.183215  ...              0.00       6
8947  C19188   23.398673  ...              0.25       6
8948  C19189   13.457564  ...              0.25       6
8949  C19190  372.708075  ...              0.00       6

[5 rows x 18 columns]

# EDA #
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8950 entries, 0 to 8949
Data columns (total 18 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   CUST_ID                           8950 non-null   object 
 1   BALANCE                           8950 non-null   float64
 2   BALANCE_FREQUENCY                 8950 non-null   float64
 3   PURCHASES                         8950 non-null   float64
 4   ONEOFF_PURCHASES                  8950 non-null   float64
 5   INSTALLMENTS_PURCHASES            8950 non-null   float64
 6   CASH_ADVANCE                      8950 non-null   float64
 7   PURCHASES_FREQUENCY               8950 non-null   float64
 8   ONEOFF_PURCHASES_FREQUENCY        8950 non-null   float64
 9   PURCHASES_INSTALLMENTS_FREQUENCY  8950 non-null   float64
 10  CASH_ADVANCE_FREQUENCY            8950 non-null   float64
 11  CASH_ADVANCE_TRX                  8950 non-null   int64  
 12  PURCHASES_TRX                     8950 non-null   int64  
 13  CREDIT_LIMIT                      8949 non-null   float64
 14  PAYMENTS                          8950 non-null   float64
 15  MINIMUM_PAYMENTS                  8637 non-null   float64
 16  PRC_FULL_PAYMENT                  8950 non-null   float64
 17  TENURE                            8950 non-null   int64  
dtypes: float64(14), int64(3), object(1)
memory usage: 1.2+ MB
df.describe()
            BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT       TENURE
count   8950.000000        8950.000000  ...       8950.000000  8950.000000
mean    1564.474828           0.877271  ...          0.153715    11.517318
std     2081.531879           0.236904  ...          0.292499     1.338331
min        0.000000           0.000000  ...          0.000000     6.000000
25%      128.281915           0.888889  ...          0.000000    12.000000
50%      873.385231           1.000000  ...          0.000000    12.000000
75%     2054.140036           1.000000  ...          0.142857    12.000000
max    19043.138560           1.000000  ...          1.000000    12.000000

[8 rows x 17 columns]
df.isnull().sum()
CUST_ID                               0
BALANCE                               0
BALANCE_FREQUENCY                     0
PURCHASES                             0
ONEOFF_PURCHASES                      0
INSTALLMENTS_PURCHASES                0
CASH_ADVANCE                          0
PURCHASES_FREQUENCY                   0
ONEOFF_PURCHASES_FREQUENCY            0
PURCHASES_INSTALLMENTS_FREQUENCY      0
CASH_ADVANCE_FREQUENCY                0
CASH_ADVANCE_TRX                      0
PURCHASES_TRX                         0
CREDIT_LIMIT                          1
PAYMENTS                              0
MINIMUM_PAYMENTS                    313
PRC_FULL_PAYMENT                      0
TENURE                                0
dtype: int64

# Filling mean value in place of missing values in the dataset #
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())
df.isnull().sum()
CUST_ID                             0
BALANCE                             0
BALANCE_FREQUENCY                   0
PURCHASES                           0
ONEOFF_PURCHASES                    0
INSTALLMENTS_PURCHASES              0
CASH_ADVANCE                        0
PURCHASES_FREQUENCY                 0
ONEOFF_PURCHASES_FREQUENCY          0
PURCHASES_INSTALLMENTS_FREQUENCY    0
CASH_ADVANCE_FREQUENCY              0
CASH_ADVANCE_TRX                    0
PURCHASES_TRX                       0
CREDIT_LIMIT                        0
PAYMENTS                            0
MINIMUM_PAYMENTS                    0
PRC_FULL_PAYMENT                    0
TENURE                              0
dtype: int64

# checking for duplicate rows in the dataset #
df.duplicated().sum()
0

# drop CUST_ID column because it is not used #
df.drop(columns=["CUST_ID"],axis=1,inplace=True)
df.head()
       BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT  TENURE
0    40.900749           0.818182  ...          0.000000      12
1  3202.467416           0.909091  ...          0.222222      12
2  2495.148862           1.000000  ...          0.000000      12
3  1666.670542           0.636364  ...          0.000000      12
4   817.714335           1.000000  ...          0.000000      12

[5 rows x 17 columns]

# Visulaization #
plt.figure(figsize=(30,45))
<Figure size 3000x4500 with 0 Axes>
for i, col in enumerate(df.columns):
       if df[col].dtype != 'object':
           ax = plt.subplot(9, 2, i+1)
           sns.kdeplot(df[col], ax=ax)
             plt.xlabel(col)
             
SyntaxError: unexpected indent
plt.show()
plt.figure(figsize=(15,15))
<Figure size 1500x1500 with 0 Axes>
sns.heatmap(df.corr(), annot=True)
<Axes: >
plt.show()

# Scaling the DataFrame #

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics
scalar=StandardScaler()
scaled_df = scalar.fit_transform(df)

# Dimensionality reduction #

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
pca_df
          PCA1      PCA2
0    -1.682220 -1.076451
1    -1.138295  2.506477
2     0.969684 -0.383520
3    -0.873628  0.043166
4    -1.599434 -0.688581
...        ...       ...
8945 -0.359629 -2.016145
8946 -0.564369 -1.639123
8947 -0.926204 -1.810786
8948 -2.336552 -0.657966
8949 -0.556422 -0.400467

[8950 rows x 2 columns]

 # Hyperparameter tuning #
 
# Finding 'k' value by Elbow Method #

inertia = []
range_val = range(1,15)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(scaled_df))
     inertia.append(kmean.inertia_)
     
plt.xlabel('Values of K')
                     
Text(0.5, 0, 'Values of K')
plt.ylabel('Inertia')
                     
Text(0, 0.5, 'Inertia')
plt.title('The Elbow Method using Inertia')
                     
Text(0.5, 1.0, 'The Elbow Method using Inertia')
plt.show()
                     
# Model Building using KMeans #
                     
kmeans_model=KMeans(4)
                     
kmeans_model.fit_predict(scaled_df)
                     
  File "C:\Users\dell i7\AppData\Roaming\Python\Python312\site-packages\joblib\externals\loky\backend\context.py", line 282, in _count_physical_cores
    raise ValueError(f"found {cpu_count_physical} physical cores < 1")
array([1, 1, 2, ..., 0, 0, 0])
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)

# Visualizing the clustered dataframe #
plt.figure(figsize=(8,8))
<Figure size 800x800 with 0 Axes>
ax=sns.scatterplot(x="PCA1",y="PCA2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
plt.title("Clustering using K-Means Algorithm")
Text(0.5, 1.0, 'Clustering using K-Means Algorithm')
plt.show()

# Find all cluster centers #

cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[df.columns])

# inverse transform the data #
cluster_centers = scalar.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers,columns=[df.columns])
cluster_centers
       BALANCE BALANCE_FREQUENCY  ... PRC_FULL_PAYMENT     TENURE
0   918.378209          0.753332  ...         0.156949   7.475610
1  2031.150707          0.863999  ...         0.037792  11.865085
2   836.179451          0.901138  ...         0.268516  11.888981
3  3867.080884          0.986689  ...         0.247215  11.940187

[4 rows x 17 columns]

# Creating a target column "Cluster" for storing the cluster segment #
                     
cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
cluster_df
          BALANCE  BALANCE_FREQUENCY  ...  TENURE  Cluster
0       40.900749           0.818182  ...      12        1
1     3202.467416           0.909091  ...      12        1
2     2495.148862           1.000000  ...      12        2
3     1666.670542           0.636364  ...      12        1
4      817.714335           1.000000  ...      12        1
...           ...                ...  ...     ...      ...
8945    28.493517           1.000000  ...       6        0
8946    19.183215           1.000000  ...       6        0
8947    23.398673           0.833333  ...       6        0
8948    13.457564           0.833333  ...       6        0
8949   372.708075           0.666667  ...       6        0

[8950 rows x 18 columns]
cluster_1_df = cluster_df[cluster_df["Cluster"]==0]
cluster_1_df
          BALANCE  BALANCE_FREQUENCY  ...  TENURE  Cluster
16    2072.074354           0.875000  ...       8        0
46     474.447149           0.500000  ...       8        0
53     464.674156           0.888889  ...       9        0
66     809.847455           0.875000  ...       8        0
72     656.013010           1.000000  ...       8        0
...           ...                ...  ...     ...      ...
8945    28.493517           1.000000  ...       6        0
8946    19.183215           1.000000  ...       6        0
8947    23.398673           0.833333  ...       6        0
8948    13.457564           0.833333  ...       6        0
8949   372.708075           0.666667  ...       6        0

[738 rows x 18 columns]
cluster_2_df = cluster_df[cluster_df["Cluster"]==1]
cluster_2_df
          BALANCE  BALANCE_FREQUENCY  ...  TENURE  Cluster
0       40.900749           0.818182  ...      12        1
1     3202.467416           0.909091  ...      12        1
3     1666.670542           0.636364  ...      12        1
4      817.714335           1.000000  ...      12        1
8     1014.926473           1.000000  ...      12        1
...           ...                ...  ...     ...      ...
8791   218.890208           0.600000  ...      10        1
8795  1478.089943           0.800000  ...      10        1
8802   427.905890           0.900000  ...      10        1
8804  1704.571464           0.800000  ...      10        1
8813  3335.053583           0.700000  ...      10        1

[4046 rows x 18 columns]
cluster_3_df = cluster_df[cluster_df["Cluster"]==2]
KeyboardInterrupt
cluster_3_df
          BALANCE  BALANCE_FREQUENCY  ...  TENURE  Cluster
2     2495.148862           1.000000  ...      12        2
5     1809.828751           1.000000  ...      12        2
7     1823.652743           1.000000  ...      12        2
10    1293.124939           1.000000  ...      12        2
12    1516.928620           1.000000  ...      12        2
...           ...                ...  ...     ...      ...
8826    33.725413           1.000000  ...       9        2
8827    30.175490           1.000000  ...       9        2
8836   112.037368           1.000000  ...       9        2
8847   224.692470           0.888889  ...       9        2
8856   227.220411           1.000000  ...       8        2

[3631 rows x 18 columns]
cluster_4_df = cluster_df[cluster_df["Cluster"] == 3]
cluster_4_df
           BALANCE  BALANCE_FREQUENCY  ...  TENURE  Cluster
6       627.260806           1.000000  ...      12        3
21     6369.531318           1.000000  ...      12        3
23     3800.151377           0.818182  ...      12        3
30    12136.219960           1.000000  ...      12        3
57     2386.330629           1.000000  ...      12        3
...            ...                ...  ...     ...      ...
8215   4436.557694           1.000000  ...      12        3
8541   3326.323283           1.000000  ...      12        3
8662    599.909949           1.000000  ...      12        3
8689    368.318662           0.909091  ...      12        3
8737   2533.618119           0.909091  ...      12        3

[535 rows x 18 columns]

# Visualization #
sns.countplot(x='Cluster', data=cluster_df)
<Axes: xlabel='Cluster', ylabel='count'>
plt.show()

for c in cluster_df.drop(['Cluster'],axis=1):
     grid= sns.FacetGrid(cluster_df, col='Cluster')
     grid= grid.map(plt.hist, c)

     
plt.show()

# Saving the kmeans clustering model and the data with cluster label #
import joblib
joblib.dump(kmeans_model, "kmeans_model.pkl")
cluster_df.to_csv("Clustered_Customer_Data.csv")

# Training and Testing the model accuracy using decision tree #

X = cluster_df.drop(['Cluster'],axis=1)
y= cluster_df[['Cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3,random_state=3)
X_train
          BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT  TENURE
5734  1079.097023                1.0  ...          0.000000      12
6631  2590.469749                1.0  ...          0.000000      12
8659   945.802599                1.0  ...          0.000000       8
8503  3074.490820                1.0  ...          0.000000      12
7433  1325.679991                1.0  ...          0.090909      12
...           ...                ...  ...               ...     ...
7161  1653.400854                1.0  ...          0.000000      11
2707   276.446975                1.0  ...          0.100000      12
6400  1434.793889                1.0  ...          0.000000      12
1688  4530.639094                1.0  ...          0.000000      12
5994   788.425212                1.0  ...          0.000000      12

[6265 rows x 17 columns]
>>> X_test
          BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT  TENURE
5535    76.673877           0.777778  ...          0.000000       9
7828   645.406005           1.000000  ...          0.000000      12
1680  4977.237859           1.000000  ...          0.000000      12
4518   380.517893           1.000000  ...          0.090909      12
6131    41.011184           1.000000  ...          0.700000      12
...           ...                ...  ...               ...     ...
265   8097.334733           1.000000  ...          0.000000      12
5034    13.221335           0.857143  ...          0.000000       7
1278  1770.072581           1.000000  ...          0.000000      12
285   4246.430225           1.000000  ...          0.083333      12
1120  4688.103434           1.000000  ...          0.000000      12

[2685 rows x 17 columns]
>>> model= DecisionTreeClassifier(criterion="entropy")
>>> model.fit(X_train, y_train)
DecisionTreeClassifier(criterion='entropy')
>>> y_pred = model.predict(X_test)
>>> print(metrics.confusion_matrix(y_test, y_pred))
[[ 206    5    2    0]
 [   6 1164   25    1]
 [  10   38 1058   17]
 [   0    6   11  136]]
>>> print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.93      0.97      0.95       213
           1       0.96      0.97      0.97      1196
           2       0.97      0.94      0.95      1123
           3       0.88      0.89      0.89       153

    accuracy                           0.95      2685
   macro avg       0.93      0.94      0.94      2685
weighted avg       0.96      0.95      0.95      2685


