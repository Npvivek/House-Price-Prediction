import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

test.head()

train.describe()

test.describe()

train.info()

test.info()

train.isnull().sum()

test.isnull().sum()

# Distribution of SalePrice
sns.histplot(train['SalePrice'], kde=True, color='blue')
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Log transformation of SalePrice to see its effect
sns.histplot(np.log1p(train['SalePrice']), kde=True, color='green')
plt.title('Log-Transformed SalePrice Distribution')
plt.xlabel('Log(SalePrice)')
plt.ylabel('Frequency')
plt.show()


# Visualizing missing data
plt.figure(figsize=(12, 8))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title('Missing Data in Training Set')
plt.show()

# Displaying the percentage of missing data for each column
missing_data = train.isnull().sum() / len(train) * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
print(missing_data)

# Visualizing missing data in the training set
missing_data = train.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
missing_percent = (missing_data / len(train)) * 100

# Plot missing data
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_percent, y=missing_percent.index, palette="viridis")
plt.title('Percentage of Missing Data by Feature')
plt.xlabel('Percentage')
plt.ylabel('Features')
plt.show()

# Function to identify low variance categorical features
def identify_low_variance_features(df, threshold=0.95):
    """
    Identify categorical features where the most common value dominates the feature
    based on a specified threshold.
    
    Parameters:
    df (DataFrame): The dataframe containing the data
    threshold (float): The threshold above which the feature will be considered low variance
    
    Returns:
    List: List of features that are considered low variance
    """
    low_variance_features = []
    
    for col in df.select_dtypes(include=['object']).columns:
        most_common_percentage = df[col].value_counts(normalize=True).max()
        if most_common_percentage > threshold:
            low_variance_features.append(col)
            print(f"Feature '{col}' is dominated by the value '{df[col].mode()[0]}' "
                  f"which appears in {most_common_percentage * 100:.2f}% of the data.")
    
    return low_variance_features

# Apply the function to the training data
low_variance_features = identify_low_variance_features(train)

print("\nLow variance features identified:", low_variance_features)


columns=['Alley','MasVnrType','FireplaceQu','PoolQC','Fence','MiscFeature','Utilities','Street']
train=train.drop(columns,axis=1)
test=test.drop(columns,axis=1)

a=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','GarageQual','GarageCond']
for i in a:
    train[i]=train[i].fillna(train[i].mode()[0])
    test[i]=test[i].fillna(test[i].mode()[0])

c=['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']
for i in c:
    test[i]=test[i].fillna(test[i].mode()[0])

b=['LotFrontage','MasVnrArea','GarageYrBlt']
for i in b:
    train[i]=train[i].fillna(train[i].mean())
    test[i]=test[i].fillna(test[i].mean())

d=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
for i in d:
    test[i]=test[i].fillna(test[i].mean())

# Select only numeric columns
numeric_cols = train.select_dtypes(include=[np.number])

# Calculate correlations with SalePrice
corr_with_target = numeric_cols.corr()['SalePrice'].sort_values(ascending=False)

# Display the top correlations
print("Top correlations with SalePrice:")
print(corr_with_target.head(20))  # Adjust the number to display as needed


# Create a subset of the correlation matrix for top features
top_features = corr_with_target.index[:20]  # Top 20 features
top_corr_matrix = numeric_cols[top_features].corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Top 20 Features')
plt.show()


# Plot the correlations with SalePrice
plt.figure(figsize=(10, 8))
sns.barplot(x=corr_with_target.head(30).values, y=corr_with_target.head(30).index, palette="viridis")
plt.title('Top 20 Features Correlated with SalePrice')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.show()


train['TotalSF']=train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF']
test['TotalSF']=test['TotalBsmtSF']+test['1stFlrSF']+test['2ndFlrSF']
train['Age']=train['YrSold']-train['YearBuilt']
test['Age']=test['YrSold']-test['YearBuilt']
train['RemodelAge']=train['YearRemodAdd']!=train['YearBuilt'].astype(int)
test['RemodelAge']=test['YearRemodAdd']!=test['YearBuilt'].astype(int)

# Relationship between TotalSF and SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train['TotalSF'], y=train['SalePrice'], hue=train['OverallQual'], palette='viridis')
plt.title('TotalSF vs SalePrice Colored by OverallQual')
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')
plt.show()

# Age of the house and SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train['Age'], y=train['SalePrice'], hue=train['OverallQual'], palette='coolwarm')
plt.title('Age vs SalePrice Colored by OverallQual')
plt.xlabel('Age of House')
plt.ylabel('SalePrice')
plt.show()

# RemodelAge and its effect
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['RemodelAge'], y=train['SalePrice'], palette='Set2')
plt.title('RemodelAge vs SalePrice')
plt.xlabel('RemodelAge (0=No, 1=Yes)')
plt.ylabel('SalePrice')
plt.show()


train=train.drop(['YearBuilt','YearRemodAdd','YrSold','TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
test=test.drop(['YearBuilt','YearRemodAdd','YrSold','TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
train['RemodelAge']=train['RemodelAge'].astype(int)
test['RemodelAge']=test['RemodelAge'].astype(int)

# Categorical feature MSZoning vs SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['MSZoning'], y=train['SalePrice'], palette='Set3')
plt.title('MSZoning vs SalePrice')
plt.xlabel('MSZoning')
plt.ylabel('SalePrice')
plt.show()

# Kitchen quality and SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['KitchenQual'], y=train['SalePrice'], palette='Set1')
plt.title('KitchenQual vs SalePrice')
plt.xlabel('KitchenQual')
plt.ylabel('SalePrice')
plt.show()

# GarageType and SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['GarageType'], y=train['SalePrice'], palette='Set2')
plt.title('GarageType vs SalePrice')
plt.xlabel('GarageType')
plt.ylabel('SalePrice')
plt.show()


le=LabelEncoder()
for i in train.columns:
    if train[i].dtype=='object':
        train[i]=le.fit_transform(train[i])
for i in test.columns:
    if test[i].dtype=='object':
        test[i]=le.fit_transform(test[i])

train['SalePrice']=np.log1p(train['SalePrice'])

x=train.drop(['SalePrice'],axis=1)
y=train['SalePrice']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state = 42)

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)
acc=rf.score(x_test,y_test)
print(acc)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)
acc=lr.score(x_test,y_test)
print(acc)

submission=pd.DataFrame()
submission['Id']=test['Id']
final_predictions=rf.predict(test)
final_predictions=np.exp(final_predictions)
submission['SalePrice']=final_predictions
submission.to_csv('submission.csv',index=False)



