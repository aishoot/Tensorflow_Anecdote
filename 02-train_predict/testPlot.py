import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np

'''
url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content
df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2 uses StringIO.StringIO
'''

df = pd.read_csv("RailwayVolume.csv")  # python2 uses StringIO.StringIO
data = np.array(df['Number(10 thousand)'])

# Normalize
normalized_data = (data - np.mean(data)) / np.std(data)
plt.figure()
plt.plot(data)
plt.savefig("history.png")
plt.show()