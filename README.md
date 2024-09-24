## EXNO-3-DS

**AIM:**
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

**ALGORITHM:**

1.Read the given Data.
2.Clean the Data Set using Data Cleaning Process.
3.Apply Feature Encoding for the feature in the data set.
4.Apply Feature Transformation for the feature in the data set.
5.Save the data to the file.

**FEATURE ENCODING**

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

**Methods Used for Data Transformation:**
  **1. FUNCTION TRANSFORMATION**
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  **2. POWER TRANSFORMATION**
• Boxcox method
• Yeojohnson method

**CODING AND OUTPUT:**
**1. FUNCTION TRANSFORMATION:**

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/64d12d6f-c5b7-439c-a793-011a28e0b432)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/232929d7-1c40-43ab-a27b-ffeb9c9a8ca3)

```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/726e44a2-15a7-45e4-99f2-7c425a37f7ba)

```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/07e30cab-02cc-4599-81af-b91d6149059c)

```
df["Highly Negative Skew"]=np.sqrt(df["Highly Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/0bc63b9d-808a-4fd4-9338-9fa043391cfe)

```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/f5b30942-2183-4c5e-8773-e7e7c3bc7650)

```
df.skew( )
```
![image](https://github.com/user-attachments/assets/7e022ba8-55e8-4f6e-9903-8bf1e763c47e)

```
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/1fa3a697-2b5a-468e-a181-341be740da25)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/8fcb1473-2b6d-48b6-acac-94324c9f17c5)

**2. POWER TRANSFORMATION:**

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8aa121f9-9ab5-4f0e-be53-a59732af88df)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4f57b40f-edf8-4bac-996a-20ef73574755)

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
![image](https://github.com/user-attachments/assets/19b0074d-511d-48e3-a583-352cc5a5ed52)

**RESULT:**
    Thus perform Feature Encoding and Transformation process has been done for the given data. 

       
