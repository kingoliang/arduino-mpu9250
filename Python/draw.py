import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("D:/Download/mxyz1.txt");
ax = df.plot.scatter(x="x",y="y",c='blue',label="XY")
ax = df.plot.scatter(x="y",y="z",c='red',label="YZ",ax=ax)
df.plot.scatter(x="x",y="z",c='green',label="XZ",ax=ax)
plt.show()