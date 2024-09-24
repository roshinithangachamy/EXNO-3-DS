## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
# 1.  FUNCTION TRANSFORMATION:

```
 import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/43db0450-1a80-4c05-854a-53d9b0cda054)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/eb220df0-7d71-45d3-b8f6-666417a74a05)

```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/014f1c39-2c94-4196-8c4d-b319bf18e273)

```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d6139ce4-099b-4f0a-88a3-b8e1fd1a48c5)

```
df["Highly Negative Skew"]=np.sqrt(df["Highly Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/ca926d56-1b07-4379-9326-5471c215a23a)

```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/09f3be86-1d85-4e3a-aec2-4f039f531f2a)

```
df.skew( )
```
![image](https://github.com/user-attachments/assets/d880e466-bd8a-4df6-b4b2-38d77f2e11ec)

```
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c1808074-7f55-433d-aff4-45a2288e3f00)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/18836dcd-f092-4b4b-96a2-c75fdc586d5a)

# 2. POWER TRANSFORMATION:
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/79013be1-c65a-44f3-bab6-16333360f978)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c4526e2d-c25d-48ca-87d4-6879559177f9)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
```
![image](https://github.com/user-attachments/assets/47b72e5a-3140-48c2-a575-d1bc2972ce5f)

# RESULT:

Thus Feature Encoding and Transformation process have been done for the given data.

       
