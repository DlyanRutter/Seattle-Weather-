import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

data_dir = '/Users/dylanrutter/Downloads/seattleweather.csv'

def get_data():
    """
    Retrieves the dataframe from source location. Replaces "DATE" category
    with either 0, indicating the date was during the spring season, 1,
    indicating summer, 2, indicating fall, 3, indicating winter, or 4,
    indicating an invalid date or no date was entered. Also replaces the
    boolean "RAIN" column with floating point 0.0 or 1.0. Returns a
    numpy array for features, a numpy array for labels, and a
    one hot array for labels
    """
    df = pd.read_csv(data_dir)
    df.drop(df.index[[21067, 18416, 18415]], inplace=True)
    
    date = df.pop('DATE')
    prcp = df.pop('PRCP')
    tmax = df.pop('TMAX')
    tmin = df.pop('TMIN')
    rain = df.pop('RAIN')
                   
    season = []    
    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    fall = ['09', '10', '11']
    winter = ['12', '01', '02']
    
    for e in date:
        month = e[5:7]
        if month in spring: season.append(0)
        elif month in summer: season.append(1)
        elif month in fall: season.append(2)
        elif month in winter: season.append(3)
        else: season.append(4)
    
    data = {'PRCP':list(prcp),
            'TMAX':list(tmax),
            'TMIN':list(tmin),
            "SEASON":season}
    
    new_df = pd.DataFrame(data=data, dtype=np.float32)
    labels = np.array(rain).astype(np.float32)
    features = new_df.as_matrix().astype(np.float32)
    one_hot = LabelBinarizer()
    bad_indices = np.where(np.isnan(features))
    if bad_indices:
        print bad_indices
    
    return features, labels, one_hot.fit_transform(labels)

def accuracy(predictions, labels):
    """
    determines accuracy
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


