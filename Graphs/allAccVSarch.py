import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

pd.set_option('expand_frame_repr', True)
data = pd.read_csv("bestResults.csv", sep = " ")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data)


# Bar graph - architecture comparison
groupLabels =['E(test_accuracy)', 'E(train_accuracy)', 'E(val_accuracy)']
groupLabelsErr = ['S(test_accuracy)','S(train_accuracy)', 'S(null)']

#updating std value
for i in range(3) :
    data.at[i,'S(test_accuracy)']= m.sqrt(data['S(test_accuracy)'][i])
    data.at[i,'S(train_accuracy)']= m.sqrt(data['S(train_accuracy)'][i])

print(data)
ax = data.plot.bar(x='arch', y=groupLabels, rot=0, yerr = 'S(test_accuracy)')
plt.show()
