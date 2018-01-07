import pandas as pd
import numpy as np
import keras, math
import tensorflow as tf


def get_dataframe():
    """
    Retrieves the dataframe from source location. Replaces "DATE" category
    with either 0, indicating the date was during the spring season, 1,
    indicating summer, 2, indicating fall, 3, indicating winter, or 4,
    indicating an invalid date or no date was entered. Also replaces the
    boolean "RAIN" column with floating point 0.0 or 1.0
    """

    df = pd.read_csv('/Users/dylanrutter/Downloads/seattleweather.csv')
    date = np.array((df.pop('DATE')))
    season = []

    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    fall = ['09', '10', '11']
    winter = ['12', '01', '02']
    
    for e in date:
        mo = e[5:7]

        if mo in spring: season.append(0)
        elif mo in summer: season.append(1)
        elif mo in fall: season.append(2)
        elif mo in winter: season.append(3)
        else: season.append(4)
    
    season = pd.Series(season)
    df["SEASON"] = season
    df.RAIN = df.RAIN.astype(float)
    
    return df
 

df = get_dataframe()
print df.head()
