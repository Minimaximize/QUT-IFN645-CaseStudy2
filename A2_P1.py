# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def data_prep():
    # Read the Model_car_sales dataset
    df = pd.read_csv('Casestudy2-Data-Py/model_car_sales.csv', na_filter=False)

    # Convert Fields HATCH, SEDAN, WAGON, UTE and K__SALES_TOT to float and set empty values to nan
    df[['HATCH','SEDAN','WAG0N','UTE','K__SALES_TOT']] =\
        df[['HATCH','SEDAN','WAG0N','UTE','K__SALES_TOT']].replace('',np.nan).astype(float)

    # df = df[df['HATCH']>0]

    return df

# if __name__=="__main__":
#     df = data_prep()
#     rs = 42
#
#     # Drop ID Value Types
#     df2 = df[['HATCH','SEDAN','WAG0N','UTE']].dropna()
#
#     # convert to Matrix
#     X = df2.as_matrix()
#
#     # Scaling
#     scalar = StandardScaler()
#     X = scalar.fit_transform(X)
#
#     # Declare lists to save clusters and costs
#     clusters = []
#     inertia_values = []
#
#     # Train clustering with a range of k values
#     for k in range(2, 15, 2):
#         model = KMeans(n_clusters=k, random_state=rs, n_jobs=10)
#         model.fit(X)
#
#         #append model to cluster list
#         clusters.append(model)
#         inertia_values.append(model.inertia_)
#
#     # plot the inertia vs K values
#     plt.plot(range(2,15,2), inertia_values, marker='*')
#     plt.show()

    # # random state
    #
    #
    # # set the random state
    # model = KMeans(n_clusters=3, random_state=rs)
    # model.fit(X)
    #
    # # Assign id to each record in X for clustering
    # y = model.predict(X)
    # df2['Clusters'] = y
    #
    # # Count records for each cluster
    # print("Cluster membership")
    # print(df2['Clusters'].value_counts())
    #
    # # pairplit the cluster distribution
    # cluster_g = sns.pairplot(df2, hue='Clusters')
    # plt.show()


    #
    # print("Sum of intra-cluster distance:", model.inertia_)
    #
    # print("Centroid locations:")
    # for centroid in model.cluster_centers_:
    #     print(centroid)
    #
    # agg_model = AgglomerativeClustering(n_clusters=3)
    # agg_model.fit(X[:50]) # subset of X, only 50 datapoints
    #


# ---------------------------------------------------------------------
"""for cols in df:
    print(df[cols].describe(),"\n")"""
#-------------------------- output -------------------------------------
# count    675.0                        # count     675
# mean     338.0                        # unique    109
# std      195.0                        # top       932
# min        1.0                        # freq       25
# 25%      169.5                        # Name: K__SALES_TOT, dtype: object
# 50%      338.0                        # Should be float?
# 75%      506.5
# max      675.0
# Name: LOCATION_NUMBER, dtype: float64
# All Values are Unique

# count            675                  # count          675
# unique             1                  # unique         675
# top       2013-04-30                  # top       Euro-565
# freq             675                  # freq             1
# Name: REPORT_DATE, dtype: object      # Name: DEALER_CODE, dtype: object
# All on the same Date                  # All unique - Good

# count     675                         # count     675
# unique    143                         # unique    518
# top                                   # top
# freq       22                         # freq       22
# Name: UTE, dtype: object              # Name: HATCH, dtype: object
# Should be Integer? 22 missing values  # Should be Integer?

# count     675                         # count     675
# unique    426                         # unique    501
# top                                   # top
# freq       22                         # freq       22
# Name: WAG0N, dtype: object            # Name: SEDAN, dtype: object
# Should be integer?                    # Should be integer?

# Count all unique Location Number Elements
"""print(df['LOCATION_NUMBER'].value_counts());"""

# Correcting Value Types
"""
for field in ['HATCH','SEDAN','WAG0N','UTE']:
    # Replace empty values with 0 for compatability
    mask = df[field] == ''
    df.loc[mask, field] = '0'

    # Convert fields into np.int64
    df[field] = df[field].astype(np.int64)

    # Change masked values to NAN
    df.loc[mask,field] = np.nan

    # Impute missing NaN's with Mean of field
    df[field].fillna(df[field].mean(), inplace=True)

    print(df[field].describe())
    print(df[field].unique())
    print("-"*20)
"""