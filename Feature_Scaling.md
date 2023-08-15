# Convert All Numberical Columns By Scaling and Convert Non-Numerical Column Into Numerical By Encoding And Display Different Types Of DataFrames

# Feature Scaling

Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values.

# NORMALIZATION
* MinMax Scaling
* Mean Scaling
* Absolute Maximum Scaling

### MinMax Scaling
* First, we are supposed to find the minimum and the maximum value of the column.
* Then we will subtract the minimum value from the entry and divide the result by the difference between the maximum and the minimum value.
* which the data will range after performing the above two steps is between 0 to 1.


```python
# Formula :                       x(old) - x(min) 
#                    X(scaled) =  ---------------
#                                 x(max) - x(min)
```

### Mean Scaling
* This method is more or less the same as the previous method but here instead of the minimum value, we subtract each entry by the mean value of the whole data and then divide the results by the difference between the minimum and the maximum value.


```python
# Formula:                        x(old) - x(MEAN)
#                   X(scaled) =   ----------------
#                                 x(max) - x(min)
```

### Absolute Maximum Scaling
* We should first select the maximum absolute value out of all the entries of a particular measure.
* Then after this, we divide each entry of the column by this maximum value.
* we will observe that each entry of the column lies in the range of -1 to 1. 
* But this method is not used that often the reason behind this is that it is too sensitive to the outliers. 


```python
# Formula:                       x(old) - Max(|x|)
#                   X(scaled) =  -----------------
#                                    Max(|x|)
```

# STANDARDIZATION
* This method scales features so that they have a mean of 0 and a standard deviation of 1
* x is the original value of the feature
* mean is the mean of the feature values
* standard deviation is the standard deviation of the feature values
* Standardization preserves the shape of the distribution and is suitable when the data doesn't have strong outliers.

Formula:                         x(old) - MEAN
                X(scaled) =   ---------------------
                                standard deviation

# ROBUST SCALING
* In this method of scaling, we use two main statistical measures of the data.
* Median , Inter-Quartile Range
* After calculating these two values we are supposed to subtract the median from each entry and then divide the result by the interquartile range.


```python
# FORMULA :                    x(old) - Q2       x(old) - x(MEDIAN)
#                X(scaled) =  -------------  =   ------------------
#                               Q3 - Q1                IQR
```

# Diamond DataSet


```python
import numpy as np
import pandas as pd
import sklearn
```


```python
# load DataSet
df = pd.read_csv(r"C:\Users\kalag\Downloads\diamonds.csv")
```


```python
df.shape
```




    (53940, 11)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['color'].unique()
```




    array(['E', 'I', 'J', 'H', 'F', 'G', 'D'], dtype=object)




```python
df['clarity'].unique()
```




    array(['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'],
          dtype=object)




```python
df.drop(columns = ['Unnamed: 0'],inplace = True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 10 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   carat    53940 non-null  float64
     1   cut      53940 non-null  object 
     2   color    53940 non-null  object 
     3   clarity  53940 non-null  object 
     4   depth    53940 non-null  float64
     5   table    53940 non-null  float64
     6   price    53940 non-null  int64  
     7   x        53940 non-null  float64
     8   y        53940 non-null  float64
     9   z        53940 non-null  float64
    dtypes: float64(6), int64(1), object(3)
    memory usage: 4.1+ MB
    


```python
df.isna().sum()
```




    carat      0
    cut        0
    color      0
    clarity    0
    depth      0
    table      0
    price      0
    x          0
    y          0
    z          0
    dtype: int64



#### No Null values are in this dataset


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>carat</th>
      <td>53940.0</td>
      <td>0.797940</td>
      <td>0.474011</td>
      <td>0.2</td>
      <td>0.40</td>
      <td>0.70</td>
      <td>1.04</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>depth</th>
      <td>53940.0</td>
      <td>61.749405</td>
      <td>1.432621</td>
      <td>43.0</td>
      <td>61.00</td>
      <td>61.80</td>
      <td>62.50</td>
      <td>79.00</td>
    </tr>
    <tr>
      <th>table</th>
      <td>53940.0</td>
      <td>57.457184</td>
      <td>2.234491</td>
      <td>43.0</td>
      <td>56.00</td>
      <td>57.00</td>
      <td>59.00</td>
      <td>95.00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>53940.0</td>
      <td>3932.799722</td>
      <td>3989.439738</td>
      <td>326.0</td>
      <td>950.00</td>
      <td>2401.00</td>
      <td>5324.25</td>
      <td>18823.00</td>
    </tr>
    <tr>
      <th>x</th>
      <td>53940.0</td>
      <td>5.731157</td>
      <td>1.121761</td>
      <td>0.0</td>
      <td>4.71</td>
      <td>5.70</td>
      <td>6.54</td>
      <td>10.74</td>
    </tr>
    <tr>
      <th>y</th>
      <td>53940.0</td>
      <td>5.734526</td>
      <td>1.142135</td>
      <td>0.0</td>
      <td>4.72</td>
      <td>5.71</td>
      <td>6.54</td>
      <td>58.90</td>
    </tr>
    <tr>
      <th>z</th>
      <td>53940.0</td>
      <td>3.538734</td>
      <td>0.705699</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>3.53</td>
      <td>4.04</td>
      <td>31.80</td>
    </tr>
  </tbody>
</table>
</div>




```python
#     spliting the data
# ----------------------------

Y = df['price']
X = df.drop(columns = 'price')
```


```python
# we split the data train and test using sklearn train-test-split

from sklearn.model_selection import train_test_split
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# From data we can sepearate catagorical and continuous

X_cont = X[['carat','depth','table','x','y','z']]
X_catg = X[['cut','color','clarity']]
```


```python
X_cont_Train,X_cont_Test,X_catg_Train,X_catg_Test,Y_Train,Y_Test = train_test_split(X_cont , X_catg , Y,
                                                                              test_size = 0.20,
                                                                              random_state = 100)
```


```python
X_cont_Train.shape,X_cont_Test.shape,X_catg_Train.shape,X_catg_Test.shape,len(Y_Train),len(Y_Test)
```




    ((43152, 6), (10788, 6), (43152, 3), (10788, 3), 43152, 10788)




```python
X_cont_Train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.01</td>
      <td>60.2</td>
      <td>59.0</td>
      <td>8.18</td>
      <td>8.12</td>
      <td>4.91</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.79</td>
      <td>62.0</td>
      <td>55.9</td>
      <td>5.88</td>
      <td>5.95</td>
      <td>3.67</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.64</td>
      <td>61.1</td>
      <td>55.0</td>
      <td>5.58</td>
      <td>5.61</td>
      <td>3.43</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.70</td>
      <td>62.7</td>
      <td>56.0</td>
      <td>5.73</td>
      <td>5.63</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.30</td>
      <td>61.6</td>
      <td>58.0</td>
      <td>4.32</td>
      <td>4.29</td>
      <td>2.65</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>1.07</td>
      <td>62.9</td>
      <td>59.0</td>
      <td>6.49</td>
      <td>6.52</td>
      <td>4.09</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.26</td>
      <td>62.6</td>
      <td>59.0</td>
      <td>4.06</td>
      <td>4.09</td>
      <td>2.55</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>0.91</td>
      <td>61.8</td>
      <td>58.0</td>
      <td>6.24</td>
      <td>6.16</td>
      <td>3.83</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>1.25</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>6.89</td>
      <td>6.85</td>
      <td>4.29</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>0.35</td>
      <td>62.1</td>
      <td>56.0</td>
      <td>4.51</td>
      <td>4.47</td>
      <td>2.79</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 6 columns</p>
</div>




```python
# copy the train data & test data 
before_Norm_Train = X_cont_Train.copy()
before_Norm_Test = X_cont_Test.copy()
```

# MinMaxScaler


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
normScaler = MinMaxScaler()
```


```python
normScaler.fit_transform(X_cont_Train)   # In the form of Array
```




    array([[0.37629938, 0.47777778, 0.30769231, 0.76163873, 0.13786078,
            0.60918114],
           [0.12266112, 0.52777778, 0.24807692, 0.54748603, 0.10101868,
            0.45533499],
           [0.09147609, 0.50277778, 0.23076923, 0.51955307, 0.09524618,
            0.42555831],
           ...,
           [0.14760915, 0.52222222, 0.28846154, 0.58100559, 0.10458404,
            0.4751861 ],
           [0.21829522, 0.53888889, 0.28846154, 0.641527  , 0.11629881,
            0.53225806],
           [0.03118503, 0.53055556, 0.25      , 0.41992551, 0.07589134,
            0.34615385]])




```python
X_cont_norm_Train = pd.DataFrame(normScaler.fit_transform(X_cont_Train),  # In the form of Pandas
                                columns = X_cont_Train.columns,
                                index = X_cont_Train.index)
```


```python
X_cont_norm_Train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.376299</td>
      <td>0.477778</td>
      <td>0.307692</td>
      <td>0.761639</td>
      <td>0.137861</td>
      <td>0.609181</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.122661</td>
      <td>0.527778</td>
      <td>0.248077</td>
      <td>0.547486</td>
      <td>0.101019</td>
      <td>0.455335</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.091476</td>
      <td>0.502778</td>
      <td>0.230769</td>
      <td>0.519553</td>
      <td>0.095246</td>
      <td>0.425558</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.103950</td>
      <td>0.547222</td>
      <td>0.250000</td>
      <td>0.533520</td>
      <td>0.095586</td>
      <td>0.441687</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.020790</td>
      <td>0.516667</td>
      <td>0.288462</td>
      <td>0.402235</td>
      <td>0.072835</td>
      <td>0.328784</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>0.180873</td>
      <td>0.552778</td>
      <td>0.307692</td>
      <td>0.604283</td>
      <td>0.110696</td>
      <td>0.507444</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.012474</td>
      <td>0.544444</td>
      <td>0.307692</td>
      <td>0.378026</td>
      <td>0.069440</td>
      <td>0.316377</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>0.147609</td>
      <td>0.522222</td>
      <td>0.288462</td>
      <td>0.581006</td>
      <td>0.104584</td>
      <td>0.475186</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>0.218295</td>
      <td>0.538889</td>
      <td>0.288462</td>
      <td>0.641527</td>
      <td>0.116299</td>
      <td>0.532258</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>0.031185</td>
      <td>0.530556</td>
      <td>0.250000</td>
      <td>0.419926</td>
      <td>0.075891</td>
      <td>0.346154</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 6 columns</p>
</div>




```python
X_cont_norm_Test = pd.DataFrame(normScaler.transform(X_cont_Test),
                                columns = X_cont_Test.columns,
                                index = X_cont_Test.index)
```


```python
X_cont_norm_Test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0.076923</td>
      <td>0.513889</td>
      <td>0.269231</td>
      <td>0.498138</td>
      <td>0.090323</td>
      <td>0.406948</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.199584</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.628492</td>
      <td>0.115620</td>
      <td>0.517370</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.064449</td>
      <td>0.561111</td>
      <td>0.288462</td>
      <td>0.470205</td>
      <td>0.086248</td>
      <td>0.397022</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.045738</td>
      <td>0.488889</td>
      <td>0.250000</td>
      <td>0.449721</td>
      <td>0.082683</td>
      <td>0.364764</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.124740</td>
      <td>0.544444</td>
      <td>0.288462</td>
      <td>0.548417</td>
      <td>0.100679</td>
      <td>0.459057</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>0.209979</td>
      <td>0.533333</td>
      <td>0.230769</td>
      <td>0.632216</td>
      <td>0.115959</td>
      <td>0.524814</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>0.286902</td>
      <td>0.502778</td>
      <td>0.307692</td>
      <td>0.692737</td>
      <td>0.127674</td>
      <td>0.566998</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>0.103950</td>
      <td>0.550000</td>
      <td>0.269231</td>
      <td>0.531657</td>
      <td>0.095416</td>
      <td>0.441687</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>0.133056</td>
      <td>0.463889</td>
      <td>0.307692</td>
      <td>0.573557</td>
      <td>0.104075</td>
      <td>0.455335</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>0.158004</td>
      <td>0.480556</td>
      <td>0.307692</td>
      <td>0.595903</td>
      <td>0.107980</td>
      <td>0.477667</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 6 columns</p>
</div>




```python
import seaborn as sns
sns.histplot(before_Norm_Train,kde = True)
```




    <Axes: ylabel='Count'>




    
![png](output_43_1.png)
    



```python
import seaborn as sns
sns.histplot(X_cont_norm_Train,kde = True)
```




    <Axes: ylabel='Count'>




    
![png](output_44_1.png)
    


# STANDARDIZATION


```python
X_cont
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53935</th>
      <td>0.72</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>0.72</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>0.70</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>0.86</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>0.75</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 6 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
```


```python
StandScaler = StandardScaler()
```


```python
X_cont_stnd_Train = pd.DataFrame(StandScaler.fit_transform(X_cont_Train),  # In the form of Pandas
                                columns = X_cont_Train.columns,
                                index = X_cont_Train.index)
```


```python
X_cont_stnd_Train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.552968</td>
      <td>-1.084836</td>
      <td>0.691895</td>
      <td>2.180549</td>
      <td>2.073410</td>
      <td>1.969676</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-0.018674</td>
      <td>0.169249</td>
      <td>-0.693121</td>
      <td>0.130733</td>
      <td>0.185285</td>
      <td>0.186833</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.334860</td>
      <td>-0.457793</td>
      <td>-1.095222</td>
      <td>-0.136634</td>
      <td>-0.110550</td>
      <td>-0.158233</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>-0.208386</td>
      <td>0.656949</td>
      <td>-0.648443</td>
      <td>-0.002950</td>
      <td>-0.093148</td>
      <td>0.028678</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-1.051547</td>
      <td>-0.109436</td>
      <td>0.245116</td>
      <td>-1.259576</td>
      <td>-1.259087</td>
      <td>-1.279699</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>0.571539</td>
      <td>0.796292</td>
      <td>0.691895</td>
      <td>0.674380</td>
      <td>0.681244</td>
      <td>0.790699</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-1.135863</td>
      <td>0.587278</td>
      <td>0.691895</td>
      <td>-1.491295</td>
      <td>-1.433108</td>
      <td>-1.423477</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>0.234274</td>
      <td>0.029906</td>
      <td>0.245116</td>
      <td>0.451574</td>
      <td>0.368007</td>
      <td>0.416877</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>0.950961</td>
      <td>0.447935</td>
      <td>0.245116</td>
      <td>1.030870</td>
      <td>0.968378</td>
      <td>1.078255</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>-0.946152</td>
      <td>0.238921</td>
      <td>-0.648443</td>
      <td>-1.090244</td>
      <td>-1.102468</td>
      <td>-1.078411</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 6 columns</p>
</div>




```python
X_cont_stnd_Test = pd.DataFrame(StandScaler.transform(X_cont_Test),
                                columns = X_cont_Test.columns,
                                index = X_cont_Test.index)
```


```python
X_cont_stnd_Test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.482413</td>
      <td>-0.179108</td>
      <td>-0.201664</td>
      <td>-0.341615</td>
      <td>-0.362880</td>
      <td>-0.373900</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.761250</td>
      <td>-0.179108</td>
      <td>-1.095222</td>
      <td>0.906098</td>
      <td>0.933574</td>
      <td>0.905721</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.608887</td>
      <td>1.005306</td>
      <td>0.245116</td>
      <td>-0.608983</td>
      <td>-0.571705</td>
      <td>-0.488922</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.798599</td>
      <td>-0.806150</td>
      <td>-0.648443</td>
      <td>-0.805052</td>
      <td>-0.754427</td>
      <td>-0.862744</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.002405</td>
      <td>0.587278</td>
      <td>0.245116</td>
      <td>0.139646</td>
      <td>0.167883</td>
      <td>0.229966</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>0.866645</td>
      <td>0.308592</td>
      <td>-1.095222</td>
      <td>0.941747</td>
      <td>0.950976</td>
      <td>0.991988</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>1.646570</td>
      <td>-0.457793</td>
      <td>0.691895</td>
      <td>1.521043</td>
      <td>1.551348</td>
      <td>1.480832</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>-0.208386</td>
      <td>0.726621</td>
      <td>-0.201664</td>
      <td>-0.020775</td>
      <td>-0.101849</td>
      <td>0.028678</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>0.086721</td>
      <td>-1.433193</td>
      <td>0.691895</td>
      <td>0.380276</td>
      <td>0.341904</td>
      <td>0.186833</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>0.339669</td>
      <td>-1.015165</td>
      <td>0.691895</td>
      <td>0.594170</td>
      <td>0.542027</td>
      <td>0.445633</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 6 columns</p>
</div>




```python
sns.histplot(before_Norm_Train,kde = True)
```




    <Axes: ylabel='Count'>




    
![png](output_53_1.png)
    



```python
sns.histplot(X_cont_stnd_Train,kde = True)
```




    <Axes: ylabel='Count'>




    
![png](output_54_1.png)
    


# Robust Scaler


```python
from sklearn.preprocessing import RobustScaler
```


```python
robstScaler = RobustScaler()
```


```python
X_cont_robst_Train = pd.DataFrame(robstScaler.fit_transform(X_cont_Train),
                                columns = X_cont_Train.columns,
                                index = X_cont_Train.index)
```


```python
X_cont_robst_Train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.046875</td>
      <td>-1.142857</td>
      <td>0.666667</td>
      <td>1.355191</td>
      <td>1.324176</td>
      <td>1.221239</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.140625</td>
      <td>0.142857</td>
      <td>-0.366667</td>
      <td>0.098361</td>
      <td>0.131868</td>
      <td>0.123894</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.093750</td>
      <td>-0.500000</td>
      <td>-0.666667</td>
      <td>-0.065574</td>
      <td>-0.054945</td>
      <td>-0.088496</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>-0.333333</td>
      <td>0.016393</td>
      <td>-0.043956</td>
      <td>0.026549</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-0.625000</td>
      <td>-0.142857</td>
      <td>0.333333</td>
      <td>-0.754098</td>
      <td>-0.780220</td>
      <td>-0.778761</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>0.578125</td>
      <td>0.785714</td>
      <td>0.666667</td>
      <td>0.431694</td>
      <td>0.445055</td>
      <td>0.495575</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.687500</td>
      <td>0.571429</td>
      <td>0.666667</td>
      <td>-0.896175</td>
      <td>-0.890110</td>
      <td>-0.867257</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>0.328125</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.295082</td>
      <td>0.247253</td>
      <td>0.265487</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>0.859375</td>
      <td>0.428571</td>
      <td>0.333333</td>
      <td>0.650273</td>
      <td>0.626374</td>
      <td>0.672566</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>-0.546875</td>
      <td>0.214286</td>
      <td>-0.333333</td>
      <td>-0.650273</td>
      <td>-0.681319</td>
      <td>-0.654867</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 6 columns</p>
</div>




```python
X_cont_robst_Test = pd.DataFrame(robstScaler.transform(X_cont_Test),
                                columns = X_cont_Test.columns,
                                index = X_cont_Test.index)
```


```python
X_cont_robst_Test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>0.796875</td>
      <td>0.285714</td>
      <td>-0.666667</td>
      <td>0.595628</td>
      <td>0.615385</td>
      <td>0.619469</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>1.375000</td>
      <td>-0.500000</td>
      <td>0.666667</td>
      <td>0.950820</td>
      <td>0.994505</td>
      <td>0.920354</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>0.000000</td>
      <td>0.714286</td>
      <td>0.000000</td>
      <td>0.005464</td>
      <td>-0.049451</td>
      <td>0.026549</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>0.218750</td>
      <td>-1.500000</td>
      <td>0.666667</td>
      <td>0.251366</td>
      <td>0.230769</td>
      <td>0.123894</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>0.406250</td>
      <td>-1.071429</td>
      <td>0.666667</td>
      <td>0.382514</td>
      <td>0.357143</td>
      <td>0.283186</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 6 columns</p>
</div>



# Categorical Encoding

# Odinal Encoding



```python
# nominal train & test
X_catg_Train_nominal = X_catg_Train[['color','clarity']]
X_catg_Test_nominal = X_catg_Test[['color','clarity']]

# odinal train & test
X_catg_Train_odinal = X_catg_Train['cut']
X_catg_Test_odinal = X_catg_Test['cut']
```


```python

```


```python
cut_dict = {'Fair':0, "Good":1,"Very Good":2,'Ideal':3,"Premium":4}
```


```python
X_catg_Train_odinal = pd.DataFrame(X_catg_Train_odinal.map(cut_dict))
```


```python
X_catg_Train_odinal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>3</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>3</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>3</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>2</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>4</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>4</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 1 columns</p>
</div>




```python
X_catg_Test_odinal = pd.DataFrame(X_catg_Test_odinal.map(cut_dict))
```


```python
X_catg_Test_odinal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>3</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>3</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>3</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>3</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>3</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>4</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>3</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 1 columns</p>
</div>



# One Hot Encoding


```python
X_catg_Train[['color','clarity']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>I</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>H</td>
      <td>VVS2</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>H</td>
      <td>VVS1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>E</td>
      <td>VS2</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>D</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>G</td>
      <td>VS2</td>
    </tr>
    <tr>
      <th>79</th>
      <td>E</td>
      <td>VVS1</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>G</td>
      <td>VVS2</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>I</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>G</td>
      <td>VVS2</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 2 columns</p>
</div>




```python
from sklearn.preprocessing import OneHotEncoder
```


```python
OneHotEncoder = OneHotEncoder(drop='first', sparse_output=False)
```


```python
X_catg_Train_Nominal_oneHot = pd.DataFrame(OneHotEncoder.fit_transform(X_catg_Train_nominal),
                                    columns = OneHotEncoder.get_feature_names_out(),
                                    index = X_catg_Train_nominal.index)
```


```python
X_catg_Train_Nominal_oneHot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 13 columns</p>
</div>




```python
X_catg_Test_Nominal_oneHot = pd.DataFrame(OneHotEncoder.transform(X_catg_Test_nominal),
                                    columns = OneHotEncoder.get_feature_names_out(),
                                    index = X_catg_Test_nominal.index)
```


```python
X_catg_Test_Nominal_oneHot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 13 columns</p>
</div>



# Binary Encoding


```python
import category_encoders
from category_encoders import BinaryEncoder
```


```python
BinaryEncoder = BinaryEncoder(cols = X_catg_Train_nominal.columns)
```


```python
X_catg_Train_Nominal_binary = pd.DataFrame(BinaryEncoder.fit_transform(X_catg_Train_nominal),
                                    columns = BinaryEncoder.get_feature_names_out(),
                                    index = X_catg_Train_nominal.index)
```


```python
X_catg_Train_Nominal_binary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 7 columns</p>
</div>




```python
X_catg_Test_Nominal_binary = pd.DataFrame(BinaryEncoder.transform(X_catg_Test_nominal),
                                        columns = BinaryEncoder.get_feature_names_out(),
                                        index = X_catg_Test_nominal.index)
```


```python
X_catg_Test_Nominal_binary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 7 columns</p>
</div>



# Target Encoding


```python
from category_encoders import TargetEncoder
```


```python
TargetEncoder = TargetEncoder(cols = ['color','clarity'])
```


```python
X_catg_Train_nominal_target = TargetEncoder.fit_transform(X_catg_Train_nominal,Y_Train)
```


```python
X_catg_Train_nominal_target
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>5077.962264</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>4481.537004</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>4481.537004</td>
      <td>2525.429945</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>3163.157295</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>4015.074074</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3086.743721</td>
      <td>2525.429945</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>4015.074074</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>5077.962264</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>4015.074074</td>
      <td>3305.480443</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 2 columns</p>
</div>




```python
X_catg_Test_nominal_target = TargetEncoder.transform(X_catg_Test_nominal,Y_Test)
```


```python
X_catg_Test_nominal_target
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>4015.074074</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>4015.074074</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>3765.883736</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>4015.074074</td>
      <td>2861.570000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>5077.962264</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>5077.962264</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>3086.743721</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>5376.169635</td>
      <td>4026.166141</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 2 columns</p>
</div>



# Leave One Out Encoder


```python
from category_encoders import LeaveOneOutEncoder
```


```python
LeaveOneOutEncoder = LeaveOneOutEncoder()
```


```python
X_catg_Train_nominal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>I</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>H</td>
      <td>VVS2</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>H</td>
      <td>VVS1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>E</td>
      <td>VS2</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>D</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>G</td>
      <td>VS2</td>
    </tr>
    <tr>
      <th>79</th>
      <td>E</td>
      <td>VVS1</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>G</td>
      <td>VVS2</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>I</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>G</td>
      <td>VVS2</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 2 columns</p>
</div>




```python
X_catg_Train_nominal_targetloe = LeaveOneOutEncoder.fit_transform(X_catg_Train_nominal,Y_Train)
```


```python
X_catg_Train_nominal_targetloe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>5075.109321</td>
      <td>4024.881234</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>4481.711148</td>
      <td>3305.475886</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>4481.897698</td>
      <td>2525.581587</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>3086.793420</td>
      <td>3940.312991</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>3163.615845</td>
      <td>4026.483088</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16304</th>
      <td>4014.795719</td>
      <td>3939.922339</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3087.066692</td>
      <td>2526.107180</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>4014.946324</td>
      <td>3305.022392</td>
    </tr>
    <tr>
      <th>14147</th>
      <td>5077.809896</td>
      <td>4026.002389</td>
    </tr>
    <tr>
      <th>38408</th>
      <td>4015.405789</td>
      <td>3306.041831</td>
    </tr>
  </tbody>
</table>
<p>43152 rows × 2 columns</p>
</div>




```python
X_catg_Test_nominal_targetloe = LeaveOneOutEncoder.transform(X_catg_Test_nominal,Y_Test)
```


```python
X_catg_Test_nominal_targetloe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>3086.819689</td>
      <td>3940.334014</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>4014.493734</td>
      <td>3863.912568</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>4015.376955</td>
      <td>4026.428148</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>3766.253345</td>
      <td>3865.176533</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>4015.046024</td>
      <td>2860.564689</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17694</th>
      <td>5077.489068</td>
      <td>3304.538386</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>5076.730955</td>
      <td>3863.733455</td>
    </tr>
    <tr>
      <th>53573</th>
      <td>3086.793420</td>
      <td>3940.312991</td>
    </tr>
    <tr>
      <th>7941</th>
      <td>3086.586713</td>
      <td>3864.660899</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>5377.108241</td>
      <td>4026.238487</td>
    </tr>
  </tbody>
</table>
<p>10788 rows × 2 columns</p>
</div>



# `Different Types of DataFrames`


```python

```

# The Dataframe of `Normalization` and `OneHotEncoding` for nominal and `OdinalEncoding` for cut column which is odinal


```python
X_Train_1 = pd.concat([X_cont_norm_Train,X_catg_Train_odinal,X_catg_Train_Nominal_oneHot],axis =1)
```


```python
X_Train_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.376299</td>
      <td>0.477778</td>
      <td>0.307692</td>
      <td>0.761639</td>
      <td>0.137861</td>
      <td>0.609181</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.122661</td>
      <td>0.527778</td>
      <td>0.248077</td>
      <td>0.547486</td>
      <td>0.101019</td>
      <td>0.455335</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.091476</td>
      <td>0.502778</td>
      <td>0.230769</td>
      <td>0.519553</td>
      <td>0.095246</td>
      <td>0.425558</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.103950</td>
      <td>0.547222</td>
      <td>0.250000</td>
      <td>0.533520</td>
      <td>0.095586</td>
      <td>0.441687</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.020790</td>
      <td>0.516667</td>
      <td>0.288462</td>
      <td>0.402235</td>
      <td>0.072835</td>
      <td>0.328784</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_1 = pd.concat([X_cont_norm_Test,X_catg_Test_odinal,X_catg_Test_Nominal_oneHot],axis =1)
```


```python
X_Test_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0.076923</td>
      <td>0.513889</td>
      <td>0.269231</td>
      <td>0.498138</td>
      <td>0.090323</td>
      <td>0.406948</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.199584</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.628492</td>
      <td>0.115620</td>
      <td>0.517370</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.064449</td>
      <td>0.561111</td>
      <td>0.288462</td>
      <td>0.470205</td>
      <td>0.086248</td>
      <td>0.397022</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.045738</td>
      <td>0.488889</td>
      <td>0.250000</td>
      <td>0.449721</td>
      <td>0.082683</td>
      <td>0.364764</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.124740</td>
      <td>0.544444</td>
      <td>0.288462</td>
      <td>0.548417</td>
      <td>0.100679</td>
      <td>0.459057</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Standadization` and `OneHotEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_2 = pd.concat([X_cont_stnd_Train,X_catg_Train_odinal,X_catg_Train_Nominal_oneHot],axis =1)
```


```python
X_Train_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.552968</td>
      <td>-1.084836</td>
      <td>0.691895</td>
      <td>2.180549</td>
      <td>2.073410</td>
      <td>1.969676</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-0.018674</td>
      <td>0.169249</td>
      <td>-0.693121</td>
      <td>0.130733</td>
      <td>0.185285</td>
      <td>0.186833</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.334860</td>
      <td>-0.457793</td>
      <td>-1.095222</td>
      <td>-0.136634</td>
      <td>-0.110550</td>
      <td>-0.158233</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>-0.208386</td>
      <td>0.656949</td>
      <td>-0.648443</td>
      <td>-0.002950</td>
      <td>-0.093148</td>
      <td>0.028678</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-1.051547</td>
      <td>-0.109436</td>
      <td>0.245116</td>
      <td>-1.259576</td>
      <td>-1.259087</td>
      <td>-1.279699</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_2 = pd.concat([X_cont_stnd_Test,X_catg_Test_odinal,X_catg_Test_Nominal_oneHot],axis =1)
```


```python
X_Test_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Robust scaling` and `OneHotEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_3 = pd.concat([X_cont_robst_Train,X_catg_Train_odinal,X_catg_Train_Nominal_oneHot],axis =1)
```


```python
X_Train_3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.046875</td>
      <td>-1.142857</td>
      <td>0.666667</td>
      <td>1.355191</td>
      <td>1.324176</td>
      <td>1.221239</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.140625</td>
      <td>0.142857</td>
      <td>-0.366667</td>
      <td>0.098361</td>
      <td>0.131868</td>
      <td>0.123894</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.093750</td>
      <td>-0.500000</td>
      <td>-0.666667</td>
      <td>-0.065574</td>
      <td>-0.054945</td>
      <td>-0.088496</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>-0.333333</td>
      <td>0.016393</td>
      <td>-0.043956</td>
      <td>0.026549</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-0.625000</td>
      <td>-0.142857</td>
      <td>0.333333</td>
      <td>-0.754098</td>
      <td>-0.780220</td>
      <td>-0.778761</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_4 = pd.concat([X_cont_robst_Test,X_catg_Test_odinal,X_catg_Test_Nominal_oneHot],axis =1)
```


```python
X_Test_4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_E</th>
      <th>color_F</th>
      <th>color_G</th>
      <th>color_H</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Normalization` and `BinaryEncoding`and`OdinalEncoding`for cut column which is odinal


```python
X_Train_5 = pd.concat([X_cont_norm_Train,X_catg_Train_odinal,X_catg_Train_Nominal_binary],axis =1)
```


```python
X_Train_5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.376299</td>
      <td>0.477778</td>
      <td>0.307692</td>
      <td>0.761639</td>
      <td>0.137861</td>
      <td>0.609181</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.122661</td>
      <td>0.527778</td>
      <td>0.248077</td>
      <td>0.547486</td>
      <td>0.101019</td>
      <td>0.455335</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.091476</td>
      <td>0.502778</td>
      <td>0.230769</td>
      <td>0.519553</td>
      <td>0.095246</td>
      <td>0.425558</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.103950</td>
      <td>0.547222</td>
      <td>0.250000</td>
      <td>0.533520</td>
      <td>0.095586</td>
      <td>0.441687</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.020790</td>
      <td>0.516667</td>
      <td>0.288462</td>
      <td>0.402235</td>
      <td>0.072835</td>
      <td>0.328784</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_5 = pd.concat([X_cont_norm_Test,X_catg_Test_odinal,X_catg_Test_Nominal_binary],axis =1)
```


```python
X_Test_5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0.076923</td>
      <td>0.513889</td>
      <td>0.269231</td>
      <td>0.498138</td>
      <td>0.090323</td>
      <td>0.406948</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.199584</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.628492</td>
      <td>0.115620</td>
      <td>0.517370</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.064449</td>
      <td>0.561111</td>
      <td>0.288462</td>
      <td>0.470205</td>
      <td>0.086248</td>
      <td>0.397022</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.045738</td>
      <td>0.488889</td>
      <td>0.250000</td>
      <td>0.449721</td>
      <td>0.082683</td>
      <td>0.364764</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.124740</td>
      <td>0.544444</td>
      <td>0.288462</td>
      <td>0.548417</td>
      <td>0.100679</td>
      <td>0.459057</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Standadization` and `BinaryEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_6 = pd.concat([X_cont_stnd_Train,X_catg_Train_odinal,X_catg_Train_Nominal_binary],axis =1)
```


```python
X_Train_6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.552968</td>
      <td>-1.084836</td>
      <td>0.691895</td>
      <td>2.180549</td>
      <td>2.073410</td>
      <td>1.969676</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-0.018674</td>
      <td>0.169249</td>
      <td>-0.693121</td>
      <td>0.130733</td>
      <td>0.185285</td>
      <td>0.186833</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.334860</td>
      <td>-0.457793</td>
      <td>-1.095222</td>
      <td>-0.136634</td>
      <td>-0.110550</td>
      <td>-0.158233</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>-0.208386</td>
      <td>0.656949</td>
      <td>-0.648443</td>
      <td>-0.002950</td>
      <td>-0.093148</td>
      <td>0.028678</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-1.051547</td>
      <td>-0.109436</td>
      <td>0.245116</td>
      <td>-1.259576</td>
      <td>-1.259087</td>
      <td>-1.279699</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_6 = pd.concat([X_cont_stnd_Test,X_catg_Test_odinal,X_catg_Test_Nominal_binary],axis =1)
```


```python
X_Test_6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.482413</td>
      <td>-0.179108</td>
      <td>-0.201664</td>
      <td>-0.341615</td>
      <td>-0.362880</td>
      <td>-0.373900</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.761250</td>
      <td>-0.179108</td>
      <td>-1.095222</td>
      <td>0.906098</td>
      <td>0.933574</td>
      <td>0.905721</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.608887</td>
      <td>1.005306</td>
      <td>0.245116</td>
      <td>-0.608983</td>
      <td>-0.571705</td>
      <td>-0.488922</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.798599</td>
      <td>-0.806150</td>
      <td>-0.648443</td>
      <td>-0.805052</td>
      <td>-0.754427</td>
      <td>-0.862744</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.002405</td>
      <td>0.587278</td>
      <td>0.245116</td>
      <td>0.139646</td>
      <td>0.167883</td>
      <td>0.229966</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Robust scaling` and `BinaryEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_7 = pd.concat([X_cont_robst_Train,X_catg_Train_odinal,X_catg_Train_Nominal_binary],axis =1)
```


```python
X_Train_7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.046875</td>
      <td>-1.142857</td>
      <td>0.666667</td>
      <td>1.355191</td>
      <td>1.324176</td>
      <td>1.221239</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.140625</td>
      <td>0.142857</td>
      <td>-0.366667</td>
      <td>0.098361</td>
      <td>0.131868</td>
      <td>0.123894</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.093750</td>
      <td>-0.500000</td>
      <td>-0.666667</td>
      <td>-0.065574</td>
      <td>-0.054945</td>
      <td>-0.088496</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>-0.333333</td>
      <td>0.016393</td>
      <td>-0.043956</td>
      <td>0.026549</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-0.625000</td>
      <td>-0.142857</td>
      <td>0.333333</td>
      <td>-0.754098</td>
      <td>-0.780220</td>
      <td>-0.778761</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_7 = pd.concat([X_cont_robst_Test,X_catg_Test_odinal,X_catg_Test_Nominal_binary],axis =1)
```


```python
X_Test_7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color_0</th>
      <th>color_1</th>
      <th>color_2</th>
      <th>clarity_0</th>
      <th>clarity_1</th>
      <th>clarity_2</th>
      <th>clarity_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Normalization` and `TargetEncoding`and`OdinalEncoding`for cut column which is odinal


```python
X_Train_8 = pd.concat([X_cont_norm_Train,X_catg_Train_odinal,X_catg_Train_nominal_target],axis =1)
```


```python
X_Train_8.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.376299</td>
      <td>0.477778</td>
      <td>0.307692</td>
      <td>0.761639</td>
      <td>0.137861</td>
      <td>0.609181</td>
      <td>4</td>
      <td>5077.962264</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.122661</td>
      <td>0.527778</td>
      <td>0.248077</td>
      <td>0.547486</td>
      <td>0.101019</td>
      <td>0.455335</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.091476</td>
      <td>0.502778</td>
      <td>0.230769</td>
      <td>0.519553</td>
      <td>0.095246</td>
      <td>0.425558</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>2525.429945</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.103950</td>
      <td>0.547222</td>
      <td>0.250000</td>
      <td>0.533520</td>
      <td>0.095586</td>
      <td>0.441687</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.020790</td>
      <td>0.516667</td>
      <td>0.288462</td>
      <td>0.402235</td>
      <td>0.072835</td>
      <td>0.328784</td>
      <td>4</td>
      <td>3163.157295</td>
      <td>4026.166141</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_8 = pd.concat([X_cont_norm_Test,X_catg_Test_odinal,X_catg_Test_nominal_target],axis =1)
```


```python
X_Test_8.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0.076923</td>
      <td>0.513889</td>
      <td>0.269231</td>
      <td>0.498138</td>
      <td>0.090323</td>
      <td>0.406948</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.199584</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.628492</td>
      <td>0.115620</td>
      <td>0.517370</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.064449</td>
      <td>0.561111</td>
      <td>0.288462</td>
      <td>0.470205</td>
      <td>0.086248</td>
      <td>0.397022</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.045738</td>
      <td>0.488889</td>
      <td>0.250000</td>
      <td>0.449721</td>
      <td>0.082683</td>
      <td>0.364764</td>
      <td>3</td>
      <td>3765.883736</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.124740</td>
      <td>0.544444</td>
      <td>0.288462</td>
      <td>0.548417</td>
      <td>0.100679</td>
      <td>0.459057</td>
      <td>4</td>
      <td>4015.074074</td>
      <td>2861.570000</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Standadization` and `TargetEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_9 = pd.concat([X_cont_stnd_Train,X_catg_Train_odinal,X_catg_Train_nominal_target],axis =1)
```


```python
X_Train_9.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.552968</td>
      <td>-1.084836</td>
      <td>0.691895</td>
      <td>2.180549</td>
      <td>2.073410</td>
      <td>1.969676</td>
      <td>4</td>
      <td>5077.962264</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-0.018674</td>
      <td>0.169249</td>
      <td>-0.693121</td>
      <td>0.130733</td>
      <td>0.185285</td>
      <td>0.186833</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.334860</td>
      <td>-0.457793</td>
      <td>-1.095222</td>
      <td>-0.136634</td>
      <td>-0.110550</td>
      <td>-0.158233</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>2525.429945</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>-0.208386</td>
      <td>0.656949</td>
      <td>-0.648443</td>
      <td>-0.002950</td>
      <td>-0.093148</td>
      <td>0.028678</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-1.051547</td>
      <td>-0.109436</td>
      <td>0.245116</td>
      <td>-1.259576</td>
      <td>-1.259087</td>
      <td>-1.279699</td>
      <td>4</td>
      <td>3163.157295</td>
      <td>4026.166141</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_9 = pd.concat([X_cont_stnd_Test,X_catg_Test_odinal,X_catg_Test_nominal_target],axis =1)
```


```python
X_Test_9.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.482413</td>
      <td>-0.179108</td>
      <td>-0.201664</td>
      <td>-0.341615</td>
      <td>-0.362880</td>
      <td>-0.373900</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.761250</td>
      <td>-0.179108</td>
      <td>-1.095222</td>
      <td>0.906098</td>
      <td>0.933574</td>
      <td>0.905721</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.608887</td>
      <td>1.005306</td>
      <td>0.245116</td>
      <td>-0.608983</td>
      <td>-0.571705</td>
      <td>-0.488922</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.798599</td>
      <td>-0.806150</td>
      <td>-0.648443</td>
      <td>-0.805052</td>
      <td>-0.754427</td>
      <td>-0.862744</td>
      <td>3</td>
      <td>3765.883736</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.002405</td>
      <td>0.587278</td>
      <td>0.245116</td>
      <td>0.139646</td>
      <td>0.167883</td>
      <td>0.229966</td>
      <td>4</td>
      <td>4015.074074</td>
      <td>2861.570000</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Robust scaling` and `TargetEncoding` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_10 = pd.concat([X_cont_robst_Train,X_catg_Train_odinal,X_catg_Train_nominal_target],axis =1)
```


```python
X_Train_10.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.046875</td>
      <td>-1.142857</td>
      <td>0.666667</td>
      <td>1.355191</td>
      <td>1.324176</td>
      <td>1.221239</td>
      <td>4</td>
      <td>5077.962264</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.140625</td>
      <td>0.142857</td>
      <td>-0.366667</td>
      <td>0.098361</td>
      <td>0.131868</td>
      <td>0.123894</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>3305.480443</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.093750</td>
      <td>-0.500000</td>
      <td>-0.666667</td>
      <td>-0.065574</td>
      <td>-0.054945</td>
      <td>-0.088496</td>
      <td>3</td>
      <td>4481.537004</td>
      <td>2525.429945</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>-0.333333</td>
      <td>0.016393</td>
      <td>-0.043956</td>
      <td>0.026549</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-0.625000</td>
      <td>-0.142857</td>
      <td>0.333333</td>
      <td>-0.754098</td>
      <td>-0.780220</td>
      <td>-0.778761</td>
      <td>4</td>
      <td>3163.157295</td>
      <td>4026.166141</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_10 = pd.concat([X_cont_robst_Test,X_catg_Test_odinal,X_catg_Test_nominal_target],axis =1)
```


```python
X_Test_10.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
      <td>3</td>
      <td>3086.743721</td>
      <td>3940.186122</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
      <td>3</td>
      <td>4015.074074</td>
      <td>4026.166141</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
      <td>3</td>
      <td>3765.883736</td>
      <td>3864.729701</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
      <td>4</td>
      <td>4015.074074</td>
      <td>2861.570000</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Normalization` and `leaveOneOutEndocing` and `OdinalEncoding`for cut column which is odinal


```python
X_Train_11 = pd.concat([X_cont_norm_Train,X_catg_Train_odinal,X_catg_Train_nominal_targetloe],axis =1)
```


```python
X_Train_11.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>0.376299</td>
      <td>0.477778</td>
      <td>0.307692</td>
      <td>0.761639</td>
      <td>0.137861</td>
      <td>0.609181</td>
      <td>4</td>
      <td>5075.109321</td>
      <td>4024.881234</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.122661</td>
      <td>0.527778</td>
      <td>0.248077</td>
      <td>0.547486</td>
      <td>0.101019</td>
      <td>0.455335</td>
      <td>3</td>
      <td>4481.711148</td>
      <td>3305.475886</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>0.091476</td>
      <td>0.502778</td>
      <td>0.230769</td>
      <td>0.519553</td>
      <td>0.095246</td>
      <td>0.425558</td>
      <td>3</td>
      <td>4481.897698</td>
      <td>2525.581587</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.103950</td>
      <td>0.547222</td>
      <td>0.250000</td>
      <td>0.533520</td>
      <td>0.095586</td>
      <td>0.441687</td>
      <td>3</td>
      <td>3086.793420</td>
      <td>3940.312991</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>0.020790</td>
      <td>0.516667</td>
      <td>0.288462</td>
      <td>0.402235</td>
      <td>0.072835</td>
      <td>0.328784</td>
      <td>4</td>
      <td>3163.615845</td>
      <td>4026.483088</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_11 = pd.concat([X_cont_norm_Test,X_catg_Test_odinal,X_catg_Test_nominal_targetloe],axis =1)
```


```python
X_Test_11.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>0.076923</td>
      <td>0.513889</td>
      <td>0.269231</td>
      <td>0.498138</td>
      <td>0.090323</td>
      <td>0.406948</td>
      <td>3</td>
      <td>3086.819689</td>
      <td>3940.334014</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.199584</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.628492</td>
      <td>0.115620</td>
      <td>0.517370</td>
      <td>3</td>
      <td>4014.493734</td>
      <td>3863.912568</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>0.064449</td>
      <td>0.561111</td>
      <td>0.288462</td>
      <td>0.470205</td>
      <td>0.086248</td>
      <td>0.397022</td>
      <td>3</td>
      <td>4015.376955</td>
      <td>4026.428148</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>0.045738</td>
      <td>0.488889</td>
      <td>0.250000</td>
      <td>0.449721</td>
      <td>0.082683</td>
      <td>0.364764</td>
      <td>3</td>
      <td>3766.253345</td>
      <td>3865.176533</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.124740</td>
      <td>0.544444</td>
      <td>0.288462</td>
      <td>0.548417</td>
      <td>0.100679</td>
      <td>0.459057</td>
      <td>4</td>
      <td>4015.046024</td>
      <td>2860.564689</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Standadization` and `leaveOneOutEndocing` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_12 = pd.concat([X_cont_stnd_Train,X_catg_Train_odinal,X_catg_Train_nominal_targetloe],axis =1)
```


```python
X_Train_12.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.552968</td>
      <td>-1.084836</td>
      <td>0.691895</td>
      <td>2.180549</td>
      <td>2.073410</td>
      <td>1.969676</td>
      <td>4</td>
      <td>5075.109321</td>
      <td>4024.881234</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-0.018674</td>
      <td>0.169249</td>
      <td>-0.693121</td>
      <td>0.130733</td>
      <td>0.185285</td>
      <td>0.186833</td>
      <td>3</td>
      <td>4481.711148</td>
      <td>3305.475886</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.334860</td>
      <td>-0.457793</td>
      <td>-1.095222</td>
      <td>-0.136634</td>
      <td>-0.110550</td>
      <td>-0.158233</td>
      <td>3</td>
      <td>4481.897698</td>
      <td>2525.581587</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>-0.208386</td>
      <td>0.656949</td>
      <td>-0.648443</td>
      <td>-0.002950</td>
      <td>-0.093148</td>
      <td>0.028678</td>
      <td>3</td>
      <td>3086.793420</td>
      <td>3940.312991</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-1.051547</td>
      <td>-0.109436</td>
      <td>0.245116</td>
      <td>-1.259576</td>
      <td>-1.259087</td>
      <td>-1.279699</td>
      <td>4</td>
      <td>3163.615845</td>
      <td>4026.483088</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_12 = pd.concat([X_cont_stnd_Test,X_catg_Test_odinal,X_catg_Test_nominal_targetloe],axis =1)
```


```python
X_Test_12.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.482413</td>
      <td>-0.179108</td>
      <td>-0.201664</td>
      <td>-0.341615</td>
      <td>-0.362880</td>
      <td>-0.373900</td>
      <td>3</td>
      <td>3086.819689</td>
      <td>3940.334014</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.761250</td>
      <td>-0.179108</td>
      <td>-1.095222</td>
      <td>0.906098</td>
      <td>0.933574</td>
      <td>0.905721</td>
      <td>3</td>
      <td>4014.493734</td>
      <td>3863.912568</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.608887</td>
      <td>1.005306</td>
      <td>0.245116</td>
      <td>-0.608983</td>
      <td>-0.571705</td>
      <td>-0.488922</td>
      <td>3</td>
      <td>4015.376955</td>
      <td>4026.428148</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.798599</td>
      <td>-0.806150</td>
      <td>-0.648443</td>
      <td>-0.805052</td>
      <td>-0.754427</td>
      <td>-0.862744</td>
      <td>3</td>
      <td>3766.253345</td>
      <td>3865.176533</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.002405</td>
      <td>0.587278</td>
      <td>0.245116</td>
      <td>0.139646</td>
      <td>0.167883</td>
      <td>0.229966</td>
      <td>4</td>
      <td>4015.046024</td>
      <td>2860.564689</td>
    </tr>
  </tbody>
</table>
</div>



# The Dataframe of `Robust scaling` and `leaveOneOutEndocing` and `OdinalEncoding` for cut column which is odinal


```python
X_Train_13 = pd.concat([X_cont_robst_Train,X_catg_Train_odinal,X_catg_Train_nominal_targetloe],axis =1)
```


```python
X_Train_13.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>2.046875</td>
      <td>-1.142857</td>
      <td>0.666667</td>
      <td>1.355191</td>
      <td>1.324176</td>
      <td>1.221239</td>
      <td>4</td>
      <td>5075.109321</td>
      <td>4024.881234</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>0.140625</td>
      <td>0.142857</td>
      <td>-0.366667</td>
      <td>0.098361</td>
      <td>0.131868</td>
      <td>0.123894</td>
      <td>3</td>
      <td>4481.711148</td>
      <td>3305.475886</td>
    </tr>
    <tr>
      <th>49238</th>
      <td>-0.093750</td>
      <td>-0.500000</td>
      <td>-0.666667</td>
      <td>-0.065574</td>
      <td>-0.054945</td>
      <td>-0.088496</td>
      <td>3</td>
      <td>4481.897698</td>
      <td>2525.581587</td>
    </tr>
    <tr>
      <th>53575</th>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>-0.333333</td>
      <td>0.016393</td>
      <td>-0.043956</td>
      <td>0.026549</td>
      <td>3</td>
      <td>3086.793420</td>
      <td>3940.312991</td>
    </tr>
    <tr>
      <th>29795</th>
      <td>-0.625000</td>
      <td>-0.142857</td>
      <td>0.333333</td>
      <td>-0.754098</td>
      <td>-0.780220</td>
      <td>-0.778761</td>
      <td>4</td>
      <td>3163.615845</td>
      <td>4026.483088</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_Test_13 = pd.concat([X_cont_robst_Test,X_catg_Test_odinal,X_catg_Test_nominal_targetloe],axis =1)
```


```python
X_Test_13.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
      <td>-0.203125</td>
      <td>-0.214286</td>
      <td>0.000000</td>
      <td>-0.191257</td>
      <td>-0.214286</td>
      <td>-0.221239</td>
      <td>3</td>
      <td>3086.819689</td>
      <td>3940.334014</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>0.718750</td>
      <td>-0.214286</td>
      <td>-0.666667</td>
      <td>0.573770</td>
      <td>0.604396</td>
      <td>0.566372</td>
      <td>3</td>
      <td>4014.493734</td>
      <td>3863.912568</td>
    </tr>
    <tr>
      <th>42161</th>
      <td>-0.296875</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>-0.355191</td>
      <td>-0.346154</td>
      <td>-0.292035</td>
      <td>3</td>
      <td>4015.376955</td>
      <td>4026.428148</td>
    </tr>
    <tr>
      <th>35974</th>
      <td>-0.437500</td>
      <td>-0.857143</td>
      <td>-0.333333</td>
      <td>-0.475410</td>
      <td>-0.461538</td>
      <td>-0.522124</td>
      <td>3</td>
      <td>3766.253345</td>
      <td>3865.176533</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>0.156250</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.103825</td>
      <td>0.120879</td>
      <td>0.150442</td>
      <td>4</td>
      <td>4015.046024</td>
      <td>2860.564689</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
