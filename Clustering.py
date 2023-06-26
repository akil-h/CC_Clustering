#!/usr/bin/env python
# coding: utf-8

# # Credit Card Clustering

# #### The aim of this project is to determine credit card user demographics via spending patterns. I wish to test and see if cardholders can be grouped into identifiable segments for business and marketing use.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import data
path = './Credit card data/CC General.csv'
cc_data = pd.read_csv(path)


# In[3]:


cc_data.head(10)


# In[4]:


cc_data.info()


# There seem to be some missing values in CREDIT_LIMIT and MINIMUM_PAYMENTS, so let's look into that!

# In[5]:


cc_data.isnull().sum().sort_values(ascending=False)


# In[8]:


# Pull all numeric features
num_features = cc_data.select_dtypes(include=np.number).columns


# In[9]:


# Let's impute data for instances with NaN values using the KNNImputer transformer
from sklearn.impute import KNNImputer
imputer = KNNImputer()
imp_data = pd.DataFrame(imputer.fit_transform(cc_data[num_features]), columns=num_features)


# Note that the dataset with imputed values only contains numeric features, leaving out CUST_ID

# In[10]:


imp_data.describe().T
# Transpose the data because there are too many columns


# # Feature Scaling
# #### Let's add a new feature on top of the ones given to us by our dataset

# In[11]:


imp_data["Credit Card Utilization Ratio"] = imp_data.BALANCE/imp_data.CREDIT_LIMIT
imp_data['Credit Card Utilization Ratio'].describe()


# #### Let's plot this out

# In[12]:


plt.hist(imp_data['Credit Card Utilization Ratio'], bins=100)
plt.title("Distribution of Credit Card Utilization Ratio")
plt.xlabel("Credit Card Utilization Ratio")
plt.ylabel("Frequency")
plt.show()


# In[13]:


# Apply a log transformation
log_data = np.log1p(imp_data['Credit Card Utilization Ratio'])

# Plot the transformed distribution
plt.hist(log_data, bins=100)
plt.title("Log Transformed Distribution of Credit Card Utilization Ratio")
plt.xlabel("Log Transformed Value")
plt.ylabel("Frequency")
plt.show()


# #### The plots and descriptive statistics above show that most cardholders have a utilization ratio below 1.5, but a general average of 0.38. This tells me that while some cardholders spend more than their allocated credit limits, most exhibit restrained purchasing habits. We want to focus on those who have higher credit utilization ratios, because they are more likely to be hit with over-limit fees and potential reductions to their credit scores.

# In[14]:


# Feature distributions
imp_data[num_features].hist(bins = 15, figsize=(20, 15), layout=(5, 4));


# # KMeans

# We want to explore some initial features that are meaningful indicators of solvency and liquidity among credit card holders. Some meaningful features include Payments, Balance, and Credit Limit simply because they provide us meaningful information about one's spending habits.

# In[15]:


test_data = imp_data[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']].rename(columns = {'BALANCE': 'Balance', 'PURCHASES': 'Purchases', 'CREDIT_LIMIT': "Credit Limit"})


# In[16]:


test_data


# ## Standardization
# 
# ### We'll need to standardize values with a range between 0 and 1 to normalize variance

# In[17]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_data = scaler.fit_transform(test_data)


# In[18]:


# Let's take a look at the scaled observations
scaled_test_data = pd.DataFrame(scaled_test_data)
scaled_test_data


# In[19]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_test_data) # assign each datapoint to one out of the five clusters


# In[20]:


test_data["Clusters"] = cluster_labels
test_data.rename(columns = {0: "Balance", 1: "Purchases", 2: "Credit Limit"}, inplace=True)


# Let's rename the clusters to identify them more easily

# In[21]:


test_data['Clusters'] = test_data['Clusters'].map({0: "Cluster 1", 1: 
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"})
test_data


# Let's plot the clustering

# In[ ]:


import plotly.graph_objects as go
PLOT = go.Figure()
for i in list(test_data["Clusters"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = test_data[test_data["Clusters"]== i]['Balance'],
                                y = test_data[test_data["Clusters"] == i]['Purchases'],
                                z = test_data[test_data["Clusters"] == i]['Credit Limit'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'Balance', titlefont_color = 'black'),
                                yaxis=dict(title = 'Purchases', titlefont_color = 'black'),
                                zaxis=dict(title = 'Credit Limit', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))


# In[6]:


# Extra code to output above
from IPython.display import display
from PIL import Image

cc_cluster_plot_png = Image.open('CC Cluster plot.png')
display(cc_cluster_plot_png)


# # PCA
# 
# ### It's important that we incorporate all important features within the cluster, and in order to bring the number of dimensions down to 3, I'll apply PCA

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

# Create the preprocessing pipeline
cluster_pipeline = Pipeline([
    ('imputer', KNNImputer()),
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components = 3, random_state=42))
])


# In[23]:


# Drop arbitrary customer ID before applying PCA
pca_dataset = cc_data.drop('CUST_ID', axis=1)


# I dropped CUST_ID because it is a non-numeric, non-ordinal feature which is assigned rather arbitrarily.

# In[24]:


pca_dataset.info()


# In[38]:


transformed_pca_dataset = cluster_pipeline.fit_transform(pca_dataset)


# To determine the optimal number of clusters, I'll use the elbow method

# In[26]:


from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans

visualizer = KElbowVisualizer(kmeans, k=(1,10))
visualizer.fit(transformed_pca_dataset)
visualizer.show()


# In[27]:


# Generate kmeans cluster labels 
kmeans = KMeans(n_clusters = 3, random_state=42) # I've initialized a fresh kmeans object with only 3 clusters
pca_cluster_labels = kmeans.fit_predict(transformed_pca_dataset)


# In[39]:


# Let's combine the X and the cluster labels
transformed_pca_dataset = pd.DataFrame(transformed_pca_dataset)
transformed_pca_dataset = pd.concat([transformed_pca_dataset,pd.DataFrame(pca_cluster_labels)], axis=1)


# In[41]:


transformed_pca_dataset.columns = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Clusters']
transformed_pca_dataset["Clusters"] = transformed_pca_dataset["Clusters"].map({0: "Dimension 1", 1: "Dimension 2", 2: "Dimension 3"})
transformed_pca_dataset


# In[43]:


import plotly.graph_objects as go
PLOT2 = go.Figure()
for i in list(transformed_pca_dataset["Clusters"].unique()):
    

    PLOT2.add_trace(go.Scatter3d(x = transformed_pca_dataset[transformed_pca_dataset["Clusters"]== i]['Dimension 1'],
                                y = transformed_pca_dataset[transformed_pca_dataset["Clusters"] == i]['Dimension 2'],
                                z = transformed_pca_dataset[transformed_pca_dataset["Clusters"] == i]['Dimension 3'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT2.update_traces(hovertemplate='Dimension 1: %{x} <br>Dimension 2 %{y} <br>Dimension 3: %{z}')

    
PLOT2.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'Dimension 1', titlefont_color = 'black'),
                                yaxis=dict(title = 'Dimension 2', titlefont_color = 'black'),
                                zaxis=dict(title = 'Dimension 3', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))


# In[4]:


# Extra code to output above
from IPython.display import display
from PIL import Image

pca_cluster_plt_png = Image.open('/Users/akilhuang/Documents/CC_Clustering/PCA Cluster plot.png')
display(pca_cluster_plt_png)

