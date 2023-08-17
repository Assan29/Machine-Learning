# Diamond Data preprocessing using Pipeline and ColumnTransformer


```python
# import libraries

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , PowerTransformer , OneHotEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```


```python
# load dataset

diamond_data = pd.read_csv(r"C:\Users\kalag\Downloads\diamonds.csv")
```


```python
diamond_data.head()
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
# droping unknown columns 

diamond_data.drop(columns = 'Unnamed: 0',inplace = True)
```


```python
diamond_data.head()
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
diamond_data.shape
```




    (53940, 10)




```python
# info of the data
diamond_data.info()
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
# checking null values

diamond_data.isna().sum()
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




```python
# basic stastistics of the data

diamond_data.describe().T
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



# Spliting The Data Into Train And Test


```python
X_Train , X_Test , Y_Train , Y_Test = train_test_split(diamond_data.drop('price',axis =1),
                                                       diamond_data['price'],
                                                       test_size = 0.20,
                                                       random_state = 100)
```


```python
X_Train.shape , X_Test.shape , len(Y_Train) , len(Y_Test)
```




    ((43152, 9), (10788, 9), 43152, 10788)




```python
diamond_data.columns
```




    Index(['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y',
           'z'],
          dtype='object')




```python
X_Train_Odinal_catg  = ['cut']
X_Train_Nominal_catg = ['color','clarity']
X_Train_continous    = ['carat','depth','table','x','y','z']
```

# Continuous Columns Pipeline


```python
cont_pipeline = Pipeline(steps = [
    
    ('SimpleImputer', SimpleImputer(strategy = 'median')),
    ('RobustScaler' , RobustScaler()),
    ('PowerTransformer' , PowerTransformer())
    
])
```

# Categorical Odinal Column Pipeline


```python
diamond_data['cut'].value_counts(normalize = True)
```




    cut
    Ideal        0.399537
    Premium      0.255673
    Very Good    0.223990
    Good         0.090953
    Fair         0.029848
    Name: proportion, dtype: float64




```python
cat_pipeline_ordinal = Pipeline(steps = [
    
    ('SimpleImputer' , SimpleImputer(strategy = 'most_frequent')),
    ('OrdinalEncoder' , OrdinalEncoder(categories = [['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']]))
    
])
```

# Categorical Nominal Column Pipeline


```python
cat_pipeline_nominal = Pipeline(steps =[
    
    ('SimpleImputer' , SimpleImputer(strategy = 'most_frequent')),
    ('OneHotEncoder' , OneHotEncoder(sparse_output = False , drop = 'first'))
    
])
```

# ColumnTransformer To Combine All The Pipelines


```python
pre_col_Transformer = ColumnTransformer(transformers = [
    ('cat_pipeline_ordinal' , cat_pipeline_ordinal , X_Train_Odinal_catg),
    ('cat_pipeline_nominal' , cat_pipeline_nominal , X_Train_Nominal_catg),
    ('cont_pipeline' , cont_pipeline , X_Train_continous),
    
],remainder = 'passthrough')
```

# One Final Pipeline


```python
final_pipeline = Pipeline(steps = [
    ('pre_col_Transformer' , pre_col_Transformer)
])
```

# Train Data Preprocessing


```python
final_pipeline.fit_transform(X_Train)   # Array out_put of Train Data
```




    array([[ 1.        ,  0.        ,  0.        , ...,  1.91252925,
             1.83624526,  1.78259616],
           [ 0.        ,  0.        ,  0.        , ...,  0.23735783,
             0.31296124,  0.27883258],
           [ 0.        ,  0.        ,  0.        , ..., -0.0339054 ,
             0.00840006, -0.06939276],
           ...,
           [ 1.        ,  0.        ,  0.        , ...,  0.54065231,
             0.48842032,  0.49722525],
           [ 1.        ,  0.        ,  0.        , ...,  1.04144837,
             1.01317964,  1.07851817],
           [ 0.        ,  0.        ,  0.        , ..., -1.14426046,
            -1.20858117, -1.12102348]])




```python
# pandas out_put of Train Data

X_Train_processed = pd.DataFrame(final_pipeline.fit_transform(X_Train) ,
                                 columns = final_pipeline.get_feature_names_out(),
                                 index = X_Train.index)           
```


```python
X_Train_processed.head()
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
      <th>cat_pipeline_ordinal__cut</th>
      <th>cat_pipeline_nominal__color_E</th>
      <th>cat_pipeline_nominal__color_F</th>
      <th>cat_pipeline_nominal__color_G</th>
      <th>cat_pipeline_nominal__color_H</th>
      <th>cat_pipeline_nominal__color_I</th>
      <th>cat_pipeline_nominal__color_J</th>
      <th>cat_pipeline_nominal__clarity_IF</th>
      <th>cat_pipeline_nominal__clarity_SI1</th>
      <th>cat_pipeline_nominal__clarity_SI2</th>
      <th>cat_pipeline_nominal__clarity_VS1</th>
      <th>cat_pipeline_nominal__clarity_VS2</th>
      <th>cat_pipeline_nominal__clarity_VVS1</th>
      <th>cat_pipeline_nominal__clarity_VVS2</th>
      <th>cont_pipeline__carat</th>
      <th>cont_pipeline__depth</th>
      <th>cont_pipeline__table</th>
      <th>cont_pipeline__x</th>
      <th>cont_pipeline__y</th>
      <th>cont_pipeline__z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27187</th>
      <td>1.0</td>
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
      <td>1.872687</td>
      <td>-1.081114</td>
      <td>0.763221</td>
      <td>1.912529</td>
      <td>1.836245</td>
      <td>1.782596</td>
    </tr>
    <tr>
      <th>3118</th>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.264667</td>
      <td>0.158309</td>
      <td>-0.662113</td>
      <td>0.237358</td>
      <td>0.312961</td>
      <td>0.278833</td>
    </tr>
    <tr>
      <th>49238</th>
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
      <td>0.0</td>
      <td>-0.119152</td>
      <td>-0.466062</td>
      <td>-1.180787</td>
      <td>-0.033905</td>
      <td>0.008400</td>
      <td>-0.069393</td>
    </tr>
    <tr>
      <th>53575</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045189</td>
      <td>0.651916</td>
      <td>-0.607288</td>
      <td>0.104072</td>
      <td>0.027100</td>
      <td>0.122581</td>
    </tr>
    <tr>
      <th>29795</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.339541</td>
      <td>-0.120638</td>
      <td>0.358524</td>
      <td>-1.362183</td>
      <td>-1.425600</td>
      <td>-1.371937</td>
    </tr>
  </tbody>
</table>
</div>



# Test Data Preprocessing


```python
final_pipeline.transform(X_Test)  # array out_put of Test Data
```




    array([[ 0.        ,  1.        ,  0.        , ..., -0.25452028,
            -0.27365404, -0.30064513],
           [ 0.        ,  0.        ,  0.        , ...,  0.9378924 ,
             0.98458125,  0.93261927],
           [ 0.        ,  0.        ,  0.        , ..., -0.5577705 ,
            -0.52187576, -0.42801054],
           ...,
           [ 0.        ,  1.        ,  0.        , ...,  0.08595299,
             0.01776223,  0.12258051],
           [ 1.        ,  1.        ,  0.        , ...,  0.47508878,
             0.46387557,  0.27883258],
           [ 1.        ,  0.        ,  0.        , ...,  0.66895839,
             0.64800224,  0.52385557]])




```python
# pandas out_put of Tast Data

X_Test_processed = pd.DataFrame(final_pipeline.transform(X_Test) ,
                                 columns = final_pipeline.get_feature_names_out(),
                                 index = X_Test.index)
```


```python
X_Test_processed.head()
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
      <th>cat_pipeline_ordinal__cut</th>
      <th>cat_pipeline_nominal__color_E</th>
      <th>cat_pipeline_nominal__color_F</th>
      <th>cat_pipeline_nominal__color_G</th>
      <th>cat_pipeline_nominal__color_H</th>
      <th>cat_pipeline_nominal__color_I</th>
      <th>cat_pipeline_nominal__color_J</th>
      <th>cat_pipeline_nominal__clarity_IF</th>
      <th>cat_pipeline_nominal__clarity_SI1</th>
      <th>cat_pipeline_nominal__clarity_SI2</th>
      <th>cat_pipeline_nominal__clarity_VS1</th>
      <th>cat_pipeline_nominal__clarity_VS2</th>
      <th>cat_pipeline_nominal__clarity_VVS1</th>
      <th>cat_pipeline_nominal__clarity_VVS2</th>
      <th>cont_pipeline__carat</th>
      <th>cont_pipeline__depth</th>
      <th>cont_pipeline__table</th>
      <th>cont_pipeline__x</th>
      <th>cont_pipeline__y</th>
      <th>cont_pipeline__z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52264</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.330152</td>
      <td>-0.189985</td>
      <td>-0.092975</td>
      <td>-0.254520</td>
      <td>-0.273654</td>
      <td>-0.300645</td>
    </tr>
    <tr>
      <th>21073</th>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.942294</td>
      <td>-0.189985</td>
      <td>-1.180787</td>
      <td>0.937892</td>
      <td>0.984581</td>
      <td>0.932619</td>
    </tr>
    <tr>
      <th>42161</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.527575</td>
      <td>1.007773</td>
      <td>0.358524</td>
      <td>-0.557770</td>
      <td>-0.521876</td>
      <td>-0.428011</td>
    </tr>
    <tr>
      <th>35974</th>
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
      <td>0.0</td>
      <td>-0.852471</td>
      <td>-0.808669</td>
      <td>-0.607288</td>
      <td>-0.790641</td>
      <td>-0.749498</td>
      <td>-0.860004</td>
    </tr>
    <tr>
      <th>7641</th>
      <td>1.0</td>
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
      <td>0.287305</td>
      <td>0.581044</td>
      <td>0.358524</td>
      <td>0.246087</td>
      <td>0.295786</td>
      <td>0.320543</td>
    </tr>
  </tbody>
</table>
</div>



# DIAGRAM PIPELINE 


```python
final_pipeline
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;pre_col_Transformer&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;cat_pipeline_ordinal&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;OrdinalEncoder&#x27;,
                                                                   OrdinalEncoder(categories=[[&#x27;Ideal&#x27;,
                                                                                               &#x27;Premium&#x27;,
                                                                                               &#x27;Very &#x27;
                                                                                               &#x27;Good&#x27;,
                                                                                               &#x27;Good&#x27;,
                                                                                               &#x27;Fair&#x27;]]))]),
                                                  [&#x27;cut&#x27;]),
                                                 (&#x27;cat_pipeline_nominal&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;OneHotEncoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 sparse_output=False))]),
                                                  [&#x27;color&#x27;, &#x27;clarity&#x27;]),
                                                 (&#x27;cont_pipeline&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;RobustScaler&#x27;,
                                                                   RobustScaler()),
                                                                  (&#x27;PowerTransformer&#x27;,
                                                                   PowerTransformer())]),
                                                  [&#x27;carat&#x27;, &#x27;depth&#x27;, &#x27;table&#x27;,
                                                   &#x27;x&#x27;, &#x27;y&#x27;, &#x27;z&#x27;])]))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;pre_col_Transformer&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;cat_pipeline_ordinal&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;OrdinalEncoder&#x27;,
                                                                   OrdinalEncoder(categories=[[&#x27;Ideal&#x27;,
                                                                                               &#x27;Premium&#x27;,
                                                                                               &#x27;Very &#x27;
                                                                                               &#x27;Good&#x27;,
                                                                                               &#x27;Good&#x27;,
                                                                                               &#x27;Fair&#x27;]]))]),
                                                  [&#x27;cut&#x27;]),
                                                 (&#x27;cat_pipeline_nominal&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;OneHotEncoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 sparse_output=False))]),
                                                  [&#x27;color&#x27;, &#x27;clarity&#x27;]),
                                                 (&#x27;cont_pipeline&#x27;,
                                                  Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;RobustScaler&#x27;,
                                                                   RobustScaler()),
                                                                  (&#x27;PowerTransformer&#x27;,
                                                                   PowerTransformer())]),
                                                  [&#x27;carat&#x27;, &#x27;depth&#x27;, &#x27;table&#x27;,
                                                   &#x27;x&#x27;, &#x27;y&#x27;, &#x27;z&#x27;])]))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label sk-toggleable__label-arrow">pre_col_Transformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;cat_pipeline_ordinal&#x27;,
                                 Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;OrdinalEncoder&#x27;,
                                                  OrdinalEncoder(categories=[[&#x27;Ideal&#x27;,
                                                                              &#x27;Premium&#x27;,
                                                                              &#x27;Very &#x27;
                                                                              &#x27;Good&#x27;,
                                                                              &#x27;Good&#x27;,
                                                                              &#x27;Fair&#x27;]]))]),
                                 [&#x27;cut&#x27;]),
                                (&#x27;cat_pipeline_nominal&#x27;,
                                 Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;OneHotEncoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                sparse_output=False))]),
                                 [&#x27;color&#x27;, &#x27;clarity&#x27;]),
                                (&#x27;cont_pipeline&#x27;,
                                 Pipeline(steps=[(&#x27;SimpleImputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;RobustScaler&#x27;,
                                                  RobustScaler()),
                                                 (&#x27;PowerTransformer&#x27;,
                                                  PowerTransformer())]),
                                 [&#x27;carat&#x27;, &#x27;depth&#x27;, &#x27;table&#x27;, &#x27;x&#x27;, &#x27;y&#x27;, &#x27;z&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-31" type="checkbox" ><label for="sk-estimator-id-31" class="sk-toggleable__label sk-toggleable__label-arrow">cat_pipeline_ordinal</label><div class="sk-toggleable__content"><pre>[&#x27;cut&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-32" type="checkbox" ><label for="sk-estimator-id-32" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-33" type="checkbox" ><label for="sk-estimator-id-33" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(categories=[[&#x27;Ideal&#x27;, &#x27;Premium&#x27;, &#x27;Very Good&#x27;, &#x27;Good&#x27;, &#x27;Fair&#x27;]])</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-34" type="checkbox" ><label for="sk-estimator-id-34" class="sk-toggleable__label sk-toggleable__label-arrow">cat_pipeline_nominal</label><div class="sk-toggleable__content"><pre>[&#x27;color&#x27;, &#x27;clarity&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-35" type="checkbox" ><label for="sk-estimator-id-35" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-36" type="checkbox" ><label for="sk-estimator-id-36" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;, sparse_output=False)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-37" type="checkbox" ><label for="sk-estimator-id-37" class="sk-toggleable__label sk-toggleable__label-arrow">cont_pipeline</label><div class="sk-toggleable__content"><pre>[&#x27;carat&#x27;, &#x27;depth&#x27;, &#x27;table&#x27;, &#x27;x&#x27;, &#x27;y&#x27;, &#x27;z&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-38" type="checkbox" ><label for="sk-estimator-id-38" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-39" type="checkbox" ><label for="sk-estimator-id-39" class="sk-toggleable__label sk-toggleable__label-arrow">RobustScaler</label><div class="sk-toggleable__content"><pre>RobustScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-40" type="checkbox" ><label for="sk-estimator-id-40" class="sk-toggleable__label sk-toggleable__label-arrow">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-41" type="checkbox" ><label for="sk-estimator-id-41" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre>[]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-42" type="checkbox" ><label for="sk-estimator-id-42" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python

```


```python

```
