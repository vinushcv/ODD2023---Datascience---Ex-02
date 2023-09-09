# Ex02-Outlier

# Aim
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them

# ALGORITHM:
## STEP 1:
Read the given Data.

## STEP 2:
Get the information about the data.

## STEP 3:
Detect the Outliers using IQR method and Z score.

## STEP 4:
Remove the outliers:

## STEP 5:
Plot the datas using box plot.

# PROGRAM:
Name:vinushcv
Reg no:212222230176

```python
import pandas as pd
import seaborn as sns
age = [1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
af
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/fa4056e2-7b10-4d56-bcb1-2eff818debdc)

```python
sns.boxplot(data=af)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/2044f74d-7ed4-48c3-9017-a86e941fe5b8)

```python
sns.scatterplot(data=af)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/25a06180-300b-4223-b9ed-5414ad898b20)

```python
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
irq=af.quantile(0.5)
low=q1-1.5*iqr
low
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/6e55f327-14b8-410d-a4af-4e07822d8bfc)

```python
high=q3+1.5*iqr
high
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/41010e4c-c24b-49f0-9d4d-2f54630bd303)

```python
aq=af[((af>=low)&(af<=high))]
aq.dropna()
```

![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/21012712-7682-4dd6-8cdc-1da66f0eadea)

```python
sns.boxplot(data=af)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/fa27a9af-a7e0-420a-8924-ba88979a1e84)


```python
af=af[((af>=low)&(af<=high))]
af.dropna()
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/65db553a-2ed4-4edd-80f0-ac571bf36cb9)

```python
sns.boxplot(data=af)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/d862a58c-05c9-48f5-b14e-df5b762f5574)

```python
sns.scatterplot(data=af)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/9fc45373-2b4d-43ec-8bcc-3b9f2f19b6f1)

```python
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
```

```python
data = {'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df=pd.DataFrame(data)
df
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/538e8fec-86d8-4c8e-8bac-75f8787b15c9)

```python
sns.boxplot(data=df)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/622262cd-3936-44f8-90c9-e2534385559b)

```python
z=np.abs(stats.zscore(df))
print(df[z['weight']>3])
```

![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/0cf02d71-eb61-4755-8c00-69f21f263200)

```python
val=[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,69,202,72,75,78,81,84,232,87,90,93,96,99,258]
df=pd.DataFrame(data)
```
```python
out=[]
def d_o(val):
  ts=3
  m=np.mean(val)
  sd=np.std(val)
  for i in val:
    z=(i-m)/sd
    if np.abs(z)>ts:
      out.append(i)
  return out
```
```python
op=d_o(val)
op
``
<img width="68" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023---Datascience---Ex-02/assets/121222763/80263fed-d9b8-4069-93d2-d194f424a419">

```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
```
```python
id=pd.read_csv("iris.csv")
id
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/a274c45f-a89e-48d8-89d0-573e0bfc7def)

```python
sns.boxplot(x='sepal_width',data=id)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/c54fe82f-785b-40cc-b684-5cdefe4209fe)

```python
c1=id.sepal_width.quantile(0.25)
c3=id.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/5fb84225-96ae-4157-8566-f831dca0e672)

```python
rid=id[((id.sepal_width<(c1-1.5*iq))|(id.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/3840bf60-b501-4cea-878f-a9bc059ecdde)

```python
delid=id[~((id.sepal_width<(c1-1.5*iq))|(id.sepal_width>(c3+1.5*iq)))]
delid
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/cbb6c5c7-587b-4a49-8fa7-6e34662039bb)

```python
sns.boxplot(x='sepal_width',data=delid)
```
![image](https://github.com/vinushcv/ODD2023---Datascience---Ex-02/assets/113975318/5401a13a-c252-48a0-893f-45aad359a9da)

# Result:
Hence the given set of data is read and the outliers are removed using the IQR method and Zscore method.




