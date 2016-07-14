
Q: What does "pay more tips" mean? How much is "more"? How you compare is to "less" tips? 

A: pay more tips is a relative term. It can be expressed as Tips paid per total amount or per distance or per duration.
More and less is also very relative. If there is a significant difference in values in the first quartile relative
to median quartile then we could say that tip was paid less. Similarly third quartile of sample points with tip_per_total
would contain high tips.

Q: What are some extreme cases.
A: Some extreme cases are when the trip duration is 24 hours or when total amount paid is negative
    
Q: Analysis per weekday, hour and number of passengers
A: Aggregating tip_per_total against weekday or hour does not have significant difference
   Regarding passgenger count. I saw a significant drop when the passenger count was 9
    
More things that can be done.
1. Run this experiment for the 12M data points
2. Use multiprocessing for Neighbourhood finder
3. In case of memory limitation drop columns from the original dataframe


```python
import numpy as np
import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import fiona
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import json
import datetime
import seaborn
%matplotlib inline 
```


```python
nyc_data = pd.read_csv('yellow_tripdata_2015-06.csv', nrows=1000000, parse_dates=[1,2])
```


```python
nyc_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RateCodeID</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1.000000e+06</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.523680</td>
      <td>1.689017</td>
      <td>1.598918e+01</td>
      <td>-72.865534</td>
      <td>40.140212</td>
      <td>1.039960</td>
      <td>-72.889442</td>
      <td>40.153767</td>
      <td>1.375050</td>
      <td>13.213712</td>
      <td>0.411108</td>
      <td>0.497761</td>
      <td>1.758864</td>
      <td>0.297774</td>
      <td>0.299720</td>
      <td>16.479824</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.499439</td>
      <td>1.331861</td>
      <td>8.707910e+03</td>
      <td>8.984223</td>
      <td>4.948283</td>
      <td>0.565749</td>
      <td>8.918407</td>
      <td>4.898203</td>
      <td>0.498638</td>
      <td>11.040034</td>
      <td>0.378762</td>
      <td>0.035974</td>
      <td>2.740201</td>
      <td>1.470553</td>
      <td>0.012264</td>
      <td>13.494211</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>-86.031525</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-736.583313</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-171.000000</td>
      <td>-1.000000</td>
      <td>-0.500000</td>
      <td>-80.000000</td>
      <td>-5.540000</td>
      <td>-0.300000</td>
      <td>-171.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>-73.991898</td>
      <td>40.735325</td>
      <td>1.000000</td>
      <td>-73.991219</td>
      <td>40.733837</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>8.800000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.700000e+00</td>
      <td>-73.981522</td>
      <td>40.752911</td>
      <td>1.000000</td>
      <td>-73.979713</td>
      <td>40.753311</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.200000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>12.350000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.190000e+00</td>
      <td>-73.966278</td>
      <td>40.768112</td>
      <td>1.000000</td>
      <td>-73.962219</td>
      <td>40.769600</td>
      <td>2.000000</td>
      <td>15.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2.450000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>18.360000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>8.000000e+06</td>
      <td>125.535568</td>
      <td>59.032299</td>
      <td>99.000000</td>
      <td>125.535568</td>
      <td>56.766262</td>
      <td>5.000000</td>
      <td>670.060000</td>
      <td>20.200000</td>
      <td>1.300000</td>
      <td>980.910000</td>
      <td>511.160000</td>
      <td>0.300000</td>
      <td>990.710000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

#### Data Columns


```python
nyc_data.dtypes
```




    VendorID                          int64
    tpep_pickup_datetime     datetime64[ns]
    tpep_dropoff_datetime    datetime64[ns]
    passenger_count                   int64
    trip_distance                   float64
    pickup_longitude                float64
    pickup_latitude                 float64
    RateCodeID                        int64
    store_and_fwd_flag               object
    dropoff_longitude               float64
    dropoff_latitude                float64
    payment_type                      int64
    fare_amount                     float64
    extra                           float64
    mta_tax                         float64
    tip_amount                      float64
    tolls_amount                    float64
    improvement_surcharge           float64
    total_amount                    float64
    dtype: object




```python
nyc_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RateCodeID</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2015-06-02 11:19:29</td>
      <td>2015-06-02 11:47:52</td>
      <td>1</td>
      <td>1.63</td>
      <td>-73.954430</td>
      <td>40.764141</td>
      <td>1</td>
      <td>N</td>
      <td>-73.974754</td>
      <td>40.754093</td>
      <td>2</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>17.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2015-06-02 11:19:30</td>
      <td>2015-06-02 11:27:56</td>
      <td>1</td>
      <td>0.46</td>
      <td>-73.971443</td>
      <td>40.758942</td>
      <td>1</td>
      <td>N</td>
      <td>-73.978539</td>
      <td>40.761909</td>
      <td>1</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2015-06-02 11:19:31</td>
      <td>2015-06-02 11:30:30</td>
      <td>1</td>
      <td>0.87</td>
      <td>-73.978111</td>
      <td>40.738434</td>
      <td>1</td>
      <td>N</td>
      <td>-73.990273</td>
      <td>40.745438</td>
      <td>1</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.20</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2015-06-02 11:19:31</td>
      <td>2015-06-02 11:39:02</td>
      <td>1</td>
      <td>2.13</td>
      <td>-73.945892</td>
      <td>40.773529</td>
      <td>1</td>
      <td>N</td>
      <td>-73.971527</td>
      <td>40.760330</td>
      <td>1</td>
      <td>13.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.86</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>17.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-06-02 11:19:32</td>
      <td>2015-06-02 11:32:49</td>
      <td>1</td>
      <td>1.40</td>
      <td>-73.979088</td>
      <td>40.776772</td>
      <td>1</td>
      <td>N</td>
      <td>-73.982162</td>
      <td>40.758999</td>
      <td>2</td>
      <td>9.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fancy data plotting.
# Shows point pick up origin density

plt.style.use = 'default' #Better Styling  
new_style = {'grid': False} #Remove grid  
matplotlib.rc('axes', **new_style)  
from matplotlib import rcParams  
rcParams['figure.figsize'] = (17.5, 17) #Size of figure  
rcParams['figure.dpi'] = 250

# P.set_axis_bgcolor('black') #Background Color

P=nyc_data.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',color='black',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=.02,alpha=.6)
P.set_axis_bgcolor('white')
```


![png](output_8_0.png)



```python
# Drop column with payment type 2 since all rows with this row have 0 tip amount
nyc_data = nyc_data.drop(nyc_data[nyc_data.payment_type == 2].index)

# Create additional features to do some EDA
nyc_data['tip_per_total'] = nyc_data['tip_amount'] / nyc_data['total_amount']
nyc_data['tip_per_distance'] = nyc_data['tip_amount'] / nyc_data['trip_distance']
nyc_data['trip_duration'] = nyc_data['tpep_dropoff_datetime'] - nyc_data['tpep_pickup_datetime']
nyc_data['trip_seconds'] = nyc_data.trip_duration.apply(lambda x: x.seconds)
nyc_data['tip_per_duration'] = nyc_data['tip_amount'] / nyc_data['trip_seconds']
nyc_data['weekday'] = nyc_data['tpep_pickup_datetime'].apply(lambda x: x.strftime("%A"))
nyc_data['pickup_hour'] = nyc_data['tpep_pickup_datetime'].apply(lambda x: x.hour)

# Drop Rows where some columns are 0 cause they will skew the results
nyc_data = nyc_data.drop(nyc_data[nyc_data.tip_per_total == 0.0].index)
nyc_data = nyc_data.drop(nyc_data[nyc_data.trip_seconds == 0.0].index)
nyc_data = nyc_data.drop(nyc_data[nyc_data.trip_distance == 0.0].index)
```


```python
weekday_hour_group = nyc_data.groupby(['weekday', 'pickup_hour']).mean()
```


```python
nyc_data.shape[0]
```




    606660




```python

```

##### EDA plots


```python
seaborn.barplot(x=nyc_data.weekday, y=nyc_data.tip_per_total)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f96a748>




![png](output_14_1.png)



```python
seaborn.barplot(x=nyc_data.pickup_hour, y=nyc_data.tip_per_total)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123a47d68>




![png](output_15_1.png)



```python
seaborn.barplot(x=nyc_data.weekday, y=nyc_data.tip_per_total)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1224208d0>




![png](output_16_1.png)



```python
seaborn.barplot(x=nyc_data.trip_distance, y=nyc_data.tip_amount)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1220e3748>




![png](output_17_1.png)



```python
seaborn.barplot(x=weekday_hour_group.weekday'=='Friday', y=weekday_hour_group.tip_per_total)
```


      File "<ipython-input-15-499d9a2da7ff>", line 1
        seaborn.barplot(x=weekday_hour_group.weekday'=='Friday', y=weekday_hour_group.tip_per_total)
                                                       ^
    SyntaxError: invalid syntax




```python
weekday_hour_group.reset_index(inplace=True)
```


```python
fg = seaborn.factorplot(x='weekday', y='tip_per_total', 
                        col='pickup_hour', data=weekday_hour_group, kind='bar')
fg.set_xlabels('')
```




    <seaborn.axisgrid.FacetGrid at 0x121c945f8>




![png](output_20_1.png)



```python
seaborn.barplot(x=nyc_data.passenger_count, y=nyc_data.tip_per_total)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x127391080>




![png](output_21_1.png)



```python

```

### Read Shape File


```python
shp = fiona.open('nycd_new/OGRGeoJSON.shp')
coords = shp.bounds
shp.close
print(coords)
w, h = coords[2] - coords[0], coords[3] - coords[1]
extra = 0.01
```

    (-74.25559136315229, 40.49613398761198, -73.70000906387122, 40.91553277700521)



```python
m = Basemap(
    projection='tmerc', ellps='WGS84',
    lon_0=np.mean([coords[0], coords[2]]),
    lat_0=np.mean([coords[1], coords[3]]),
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - (extra * h), 
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + (extra * h))

_out = m.readshapefile('nycd_new/OGRGeoJSON', name='nyc', drawbounds=False, color='none', zorder=2)
```


```python
def get_town_name(lon, lat):
    point = Point(m(lon,lat))
    found = 0 
    for index, row in df_map.iterrows():
        name, polygon = row
        if polygon.contains(point):
            return name
            found = 1
            
    if not found:
        return np.NaN
```

#### Find Borough City Names for all rows in the Dataframe
#### This takes a lot of time


```python
# set up a map dataframe
df_map = pd.DataFrame({
    'poly': [Polygon(town_points) for town_points in m.nyc],
    'name': [town['neighborho'] for town in m.nyc_info]
})
```


```python
nyc_data['Neighbourhood'] = nyc_data.apply(lambda row: get_town_name(row['pickup_longitude'], row['pickup_latitude']), axis=1)
```


```python
nyc_data.Neighbourhood.unique()
```




    array(['Midtown', 'Kips Bay', 'Upper East Side', 'Chelsea',
           'LaGuardia Airport', 'Upper West Side', 'Stuyvesant Town',
           'Murray Hill', 'West Village', 'Tribeca', 'NoHo', 'East Harlem',
           'SoHo', 'Gramercy', 'Harlem', "Hell's Kitchen", 'Theater District',
           'Central Park', 'John F. Kennedy International Airport',
           'Flatiron District', 'Battery Park City', 'Financial District',
           'East Village', 'Greenwich Village', 'Chinatown', 'Civic Center',
           nan, 'Little Italy', 'Nolita', 'East Elmhurst', 'Astoria',
           'Long Island City', 'Downtown Brooklyn', 'Morningside Heights',
           'Lower East Side', 'Washington Heights', 'Bedford-Stuyvesant',
           'DUMBO', 'Rosedale', 'Red Hook', 'Two Bridges', 'Fort Greene',
           'Cobble Hill', 'Williamsburg', 'South Ozone Park', 'Briarwood',
           'Park Slope', 'Brooklyn Heights', 'Prospect Heights',
           'Windsor Terrace', 'Jackson Heights', 'Bushwick', 'South Slope',
           'Sunnyside', 'Kew Gardens', 'Kensington', 'Gowanus', 'Greenpoint',
           "Randall's Island", 'Boerum Hill', 'Crown Heights', 'Parkchester',
           'Ditmars Steinway', 'Midwood', 'Jamaica Hills', 'Flushing',
           'Cypress Hills', 'Clinton Hill', 'Inwood', 'Port Morris',
           'Woodside', 'Mount Eden', 'Sunset Park', 'Elmhurst', 'Longwood',
           'Prospect-Lefferts Gardens', 'Jamaica',
           'Flushing Meadows Corona Park', 'Concourse', 'Corona', 'Maspeth',
           'Mott Haven', 'Carroll Gardens', 'Castle Hill', 'Richmond Hill',
           'Springfield Gardens', 'Rego Park', 'Flatbush', 'East Flatbush',
           'Woodhaven', 'Bay Ridge', 'Manhattan Beach', 'Prospect Park',
           'Columbia St', 'Middle Village', 'Forest Hills',
           'Concourse Village', 'Highbridge', 'Roosevelt Island', 'Ridgewood',
           'St. Albans', 'Brownsville', 'East New York', 'Fort Hamilton',
           'Ferry Point Park', 'Navy Yard', 'Kingsbridge', 'Tremont',
           'Pelham Gardens', 'Clason Point', 'Borough Park', 'Vinegar Hill',
           'Fordham', 'Canarsie', 'Kew Gardens Hills', 'Sheepshead Bay',
           'Gravesend', 'Forest Park', 'Morrisania', 'Bensonhurst',
           'College Point', 'Queens Village', 'Green-Wood Cemetery',
           'Westchester Square', 'East Morrisania', 'Marble Hill',
           'New Springville', 'Ozone Park', 'Flatlands', 'Fresh Meadows',
           'Unionport', 'Bath Beach', 'Norwood', 'Melrose', 'Dyker Heights',
           'Jamaica Estates', 'Van Nest', 'Allerton', 'Bronxdale',
           'Howard Beach', 'Bellerose', 'Mount Hope', 'Throgs Neck',
           'Glendale', 'Spuyten Duyvil', 'Bayside', 'Claremont Village',
           'Hollis', 'Morris Heights', 'Woodlawn', 'University Heights',
           'Castleton Corners'], dtype=object)




```python
nyc_data.to_csv('nyc_tip_with_neighbourhood.csv')
```

#### Group data by Neighbour Hoods


```python
neighbourhood_group_data = nyc_data.groupby('Neighbourhood').median()
```


```python
neighbourhood_group_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RateCodeID</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>...</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>tip_per_total</th>
      <th>tip_per_distance</th>
      <th>trip_seconds</th>
      <th>tip_per_duration</th>
      <th>pickup_hour</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Allerton</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.210</td>
      <td>-73.868263</td>
      <td>40.858112</td>
      <td>1.0</td>
      <td>-73.888908</td>
      <td>40.860508</td>
      <td>1.0</td>
      <td>6.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>9.30</td>
      <td>0.161290</td>
      <td>1.239669</td>
      <td>297.0</td>
      <td>0.005051</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>Astoria</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.190</td>
      <td>-73.922531</td>
      <td>40.766022</td>
      <td>1.0</td>
      <td>-73.931839</td>
      <td>40.758789</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.5</td>
      <td>2.56</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>16.80</td>
      <td>0.166667</td>
      <td>0.973684</td>
      <td>745.0</td>
      <td>0.003843</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Bath Beach</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>10.800</td>
      <td>-74.005775</td>
      <td>40.603100</td>
      <td>1.0</td>
      <td>-74.013130</td>
      <td>40.707500</td>
      <td>1.0</td>
      <td>36.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>8.00</td>
      <td>5.54</td>
      <td>0.3</td>
      <td>50.84</td>
      <td>0.157356</td>
      <td>0.740741</td>
      <td>2615.0</td>
      <td>0.003059</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Battery Park City</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.490</td>
      <td>-74.015572</td>
      <td>40.713905</td>
      <td>1.0</td>
      <td>-73.992599</td>
      <td>40.739334</td>
      <td>1.0</td>
      <td>14.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>2.56</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>18.35</td>
      <td>0.166667</td>
      <td>0.895762</td>
      <td>913.0</td>
      <td>0.003199</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Bay Ridge</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.625</td>
      <td>-74.025826</td>
      <td>40.629696</td>
      <td>1.0</td>
      <td>-74.000507</td>
      <td>40.627218</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>...</td>
      <td>0.5</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>18.65</td>
      <td>0.157912</td>
      <td>0.816667</td>
      <td>849.5</td>
      <td>0.003286</td>
      <td>14.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
neighbourhood_group_data.reset_index(inplace=True)
agg_df = neighbourhood_group_data[['Neighbourhood', 'tip_per_total', 'tip_per_distance', 'tip_per_duration']]
```

#### Create Geo DF to be merged with aggregated Data Frame


```python
geo_df = pd.DataFrame(m.nyc_info)
geo_df_neighbourhoods = geo_df[['neighborho', 'borough', 'boroughCod']]
```


```python
agg_geo_df = pd.merge(left=geo_df_neighbourhoods, right=agg_df, how='left', left_on='neighborho', right_on='Neighbourhood')
```

#### Create Map for different metrics


```python
import folium
```


```python
nyc_geo = 'nyc_new.geojson'
map = folium.Map(location=[40.730610, -73.935242], zoom_start=6)
map.geo_json(geo_path=nyc_geo, data=agg_geo_df,
             columns=['neighborho', 'tip_per_total'],
             key_on='feature.properties.neighborhood',
#              threshold_scale=[0.0, 0.156, 0.157, 0.158, 0.159],
             fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
             legend_name='NYC Tip Anaysis')
map.save('nyc_tip_per_total.html')

```

    /Users/vaibhav.singh/.virtualenvs/py3/lib/python3.5/site-packages/folium/folium.py:504: UserWarning: This method is deprecated. Please use Map.choropleth instead.
      warnings.warn('This method is deprecated. '
    /Users/vaibhav.singh/.virtualenvs/py3/lib/python3.5/site-packages/folium/folium.py:506: FutureWarning: 'threshold_scale' default behavior has changed. Now you get a linear scale between the 'min' and the 'max' of your data. To get former behavior, use folium.utilities.split_six.
      return self.choropleth(*args, **kwargs)



```python
map = folium.Map(location=[40.730610, -73.935242], zoom_start=6)
map.choropleth(geo_path=nyc_geo, data=agg_geo_df,
             columns=['neighborho', 'tip_per_distance'],
             key_on='feature.properties.neighborhood',
#              threshold_scale=[0.0, 0.5, 0.1, 0.15, 0.2],
             fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
             legend_name='NYC Tip Anaysis')
map.save('nyc_tip_per_distance.html')
```

    /Users/vaibhav.singh/.virtualenvs/py3/lib/python3.5/site-packages/ipykernel/__main__.py:7: FutureWarning: 'threshold_scale' default behavior has changed. Now you get a linear scale between the 'min' and the 'max' of your data. To get former behavior, use folium.utilities.split_six.



```python
map = folium.Map(location=[40.730610, -73.935242], zoom_start=6)
map.choropleth(geo_path=nyc_geo, data=agg_geo_df,
             columns=['neighborho', 'tip_per_duration'],
             key_on='feature.properties.neighborhood',
#              threshold_scale=[0.0, 0.5, 0.1, 0.15, 0.2],
             fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
             legend_name='NYC Tip Anaysis')
map.save('nyc_tip_per_duration.html')
```

    /Users/vaibhav.singh/.virtualenvs/py3/lib/python3.5/site-packages/ipykernel/__main__.py:7: FutureWarning: 'threshold_scale' default behavior has changed. Now you get a linear scale between the 'min' and the 'max' of your data. To get former behavior, use folium.utilities.split_six.



```python
from IPython.display import IFrame
IFrame('nyc_tip_per_total.html', width=700, height=350)
```





        <iframe
            width="700"
            height="350"
            src="nyc_tip_per_total.html"
            frameborder="0"
            allowfullscreen
        ></iframe>
        




```python
IFrame('nyc_tip_per_distance.html', width=700, height=350)
```





        <iframe
            width="700"
            height="350"
            src="nyc_tip_per_distance.html"
            frameborder="0"
            allowfullscreen
        ></iframe>
        




```python
IFrame('nyc_tip_per_duration.html', width=700, height=350)
```





        <iframe
            width="700"
            height="350"
            src="nyc_tip_per_duration.html"
            frameborder="0"
            allowfullscreen
        ></iframe>
        



### Conclusions


```python
agg_geo_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighborho</th>
      <th>borough</th>
      <th>boroughCod</th>
      <th>Neighbourhood</th>
      <th>tip_per_total</th>
      <th>tip_per_distance</th>
      <th>tip_per_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allerton</td>
      <td>Bronx</td>
      <td>2</td>
      <td>Allerton</td>
      <td>0.16129</td>
      <td>1.239669</td>
      <td>0.005051</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alley Pond Park</td>
      <td>Queens</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arden Heights</td>
      <td>Staten Island</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arlington</td>
      <td>Staten Island</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arrochar</td>
      <td>Staten Island</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



##### Highest tip paying neighbourhood ordered by tip_per_total


```python
agg_geo_df.sort_values(['tip_per_total'], ascending=False).iloc[0]
```




    neighborho          Queens Village
    borough                     Queens
    boroughCod                       4
    Neighbourhood       Queens Village
    tip_per_total             0.244634
    tip_per_distance           1.13275
    tip_per_duration        0.00672607
    Name: 244, dtype: object



##### Highest tip paying neighbourhood ordered by tip_per_distance


```python
agg_geo_df.sort_values(['tip_per_distance'], ascending=False).iloc[0]
```




    neighborho           Van Nest
    borough                 Bronx
    boroughCod                  2
    Neighbourhood        Van Nest
    tip_per_total        0.196472
    tip_per_distance      1.70324
    tip_per_duration    0.0117337
    Name: 292, dtype: object



##### Highest tip paying neighbourhood ordered by tip_per_duration


```python
agg_geo_df.sort_values(['tip_per_duration'], ascending=False).iloc[0]
```




    neighborho          Woodhaven
    borough                Queens
    boroughCod                  4
    Neighbourhood       Woodhaven
    tip_per_total        0.198581
    tip_per_distance     0.947141
    tip_per_duration     0.129206
    Name: 306, dtype: object



##### Aggregating information on Borough


```python
agg_geo_df.groupby(['borough']).median().head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tip_per_total</th>
      <th>tip_per_distance</th>
      <th>tip_per_duration</th>
    </tr>
    <tr>
      <th>borough</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bronx</th>
      <td>0.166667</td>
      <td>1.006742</td>
      <td>0.003741</td>
    </tr>
    <tr>
      <th>Brooklyn</th>
      <td>0.166667</td>
      <td>0.982648</td>
      <td>0.003519</td>
    </tr>
    <tr>
      <th>Manhattan</th>
      <td>0.166667</td>
      <td>1.111636</td>
      <td>0.003175</td>
    </tr>
    <tr>
      <th>Queens</th>
      <td>0.166667</td>
      <td>0.947581</td>
      <td>0.004075</td>
    </tr>
    <tr>
      <th>Staten Island</th>
      <td>0.117035</td>
      <td>0.470615</td>
      <td>0.003123</td>
    </tr>
  </tbody>
</table>
</div>


