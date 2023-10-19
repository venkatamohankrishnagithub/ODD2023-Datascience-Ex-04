# Ex04: Multivariate AnalysisAssignment
# AIM
To perform Multivariate EDA on the given data set.
# EXPLANATION
- <B>Exploratory data analysis</B> is used to understand the messages within a dataset.<br>
- This technique involves many iterative processes to ensure that the cleaned data is further sorted to better understand the useful meaning.<br>
- The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
# ALGORITHM
### STEP 1
Import the built libraries required to perform EDA and outlier removal.
### STEP 2
Read the given csv file
### STEP 3
Convert the file into a dataframe and get information of the data.
### STEP 4
Return the objects containing counts of unique values using (value_counts()).
### STEP 5
Plot the counts in the form of Histogram or Bar Graph.
### STEP 6
Use seaborn the bar graph comparison of data can be viewed.
### STEP 7
Find the pairwise correlation of all columns in the dataframe.corr()
### STEP 8
Save the final data set into the file

# CODE
- <B>Diabetes.csv</B>
```python
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
import matplotlib.pyplot as plt
df = pd.read_csv("/content/diabetes.csv")
df.info()
df.isnull().sum()
sb.boxplot(data=df)

# DATA CLEANING
z = np.abs(stats.zscore(df['Glucose']))
dfc=df[(z<2)]
z = np.abs(stats.zscore(dfc['BloodPressure']))
dfc=dfc[(z<2)]
z = np.abs(stats.zscore(dfc['SkinThickness']))
dfc=dfc[(z<3)]
z = np.abs(stats.zscore(dfc['BMI']))
dfc=dfc[(z<2)]
z = np.abs(stats.zscore(dfc['Insulin']))
dfc=dfc[(z<2)]
z = np.abs(stats.zscore(dfc['DiabetesPedigreeFunction']))
dfc=dfc[(z<2)]
z = np.abs(stats.zscore(dfc['Age']))
dfc=dfc[(z<2)]
z = np.abs(stats.zscore(dfc['Outcome']))
dfc=dfc[(z<3)]

sb.boxplot(data=dfc)
plt.figure(figsize = (14,6))
sb.scatterplot(x = 'Glucose',y='BloodPressure',data = df)
sb.scatterplot(x = 'Glucose',y='Insulin',data = df)
sb.scatterplot(x = 'Glucose',y='DiabetesPedigreeFunction',data = df)
sb.scatterplot(x = 'Glucose',y='Age',data = df)
sb.heatmap(df.corr(),annot = True)
```
- <B>SuperStore.csv</B>
```python
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
import matplotlib.pyplot as plt
df = pd.read_csv("/content/SuperStore.csv")
df.info()
df.isnull().sum()

# FILLING NULL VALUES
df['Postal Code'] = df['Postal Code'].fillna(value=df['Postal Code'].mode()[0])
df.isnull().sum()
sb.boxplot(data=df)

# DATA CLEANING
z = np.abs(stats.zscore(df['Sales']))
df = df[z<3]
sb.boxplot(data=df['Sales'])
sb.scatterplot(x = 'Postal Code',y='Sales',data = df)
sb.scatterplot(x = 'Row ID',y='Sales',data = df)
sb.heatmap(df.corr(),annot = True)
```
# OUTPUT
- <B>Diabetes.csv</B>

<br><img src="https://github.com/Janarthanan2/ODD2023-DataScience-Ex-03/assets/119393515/91f8d1c2-9cea-4360-9647-b4a200b5eafe" width="250" height="250">

<br><img src="https://github.com/Janarthanan2/ODD2023-DataScience-Ex-03/assets/119393515/77a451e8-4b53-410a-a88d-9b3ec16779e8" width="250" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/b55fb781-1b97-42e6-a22e-2258f2e40430" width="350" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/f054c181-deee-4f3f-a9da-c0a20947ba2d" width="250" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/6203d6c5-b03a-4f97-a35d-3cc87acbbe32" width="250" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/66991054-33ae-4eae-b03c-bdfcf0f7033e" width="250" height="250"><br>

<br><img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/8b329a1d-fce5-463f-8b26-a24725c8ea84" width="350" height="350">

- <B>SuperStore.csv</B>

<br><img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/0f1b50e5-aec8-4350-8636-b4b0b7197a30" width="250" height="250">

<br><img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/c257b8c8-a8f7-4aa1-9080-d67f702fb20d" width="250" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/270781dd-41f1-4d07-ba56-b5fd4da20970" width="250" height="250">
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/b12eb5aa-27e1-4076-991f-f89cc23ea7fc" width="250" height="250"><br>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-04/assets/119393515/416c5763-61e1-4082-bcfa-bb47d6e9ed18" width="250" height="250">

# RESULT
Thus we have read the given data and performed the multivariate analysis with different types of plots.
