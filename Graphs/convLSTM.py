"""
File description : convergence analysis ----------------------------------------
                   For each input data folder, builds the mean accuracy (test)
                   array, then computes the associated std per epoch

"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x1 = []

y001 = [0.5401347875595093, 0.6632965803146362, 0.6691176295280457, 0.6792279481887817, 0.6801470518112183, 0.6663603186607361, 0.6850489974021912, 0.6832107901573181, 0.6847426295280457, 0.6933210492134094, 0.6948529481887817, 0.6948529481887817, 0.6914828419685364, 0.689338207244873, 0.6924019455909729, 0.6942402124404907, 0.6902573704719543, 0.7000612616539001, 0.6927083134651184, 0.6945465803146362, 0.6979166865348816, 0.7015931606292725, 0.6920955777168274, 0.6979166865348816, 0.6927083134651184, 0.6948529481887817, 0.6927083134651184, 0.6979166865348816, 0.6976103186607361, 0.7055760025978088, 0.6957720518112183, 0.6973039507865906, 0.6982230544090271, 0.6973039507865906, 0.6939338445663452, 0.6988357901573181, 0.6951593160629272, 0.6908701062202454, 0.7040441036224365, 0.6966911554336548, 0.6991421580314636, 0.6957720518112183, 0.701899528503418, 0.688725471496582, 0.6890318393707275, 0.686274528503418, 0.7015931606292725, 0.6914828419685364, 0.6991421580314636, 0.6945465803146362, 0.6997548937797546, 0.7006739974021912, 0.7006739974021912, 0.7052696347236633, 0.689338207244873, 0.6969975233078003, 0.6973039507865906, 0.703737735748291, 0.702512264251709, 0.6973039507865906, 0.6902573704719543, 0.6957720518112183, 0.6991421580314636, 0.7046568393707275, 0.6951593160629272, 0.7028186321258545, 0.7061887383460999, 0.7003676295280457, 0.6957720518112183, 0.6985294222831726, 0.6973039507865906, 0.6948529481887817, 0.6988357901573181, 0.7034313678741455, 0.6976103186607361, 0.6988357901573181, 0.7015931606292725, 0.6997548937797546, 0.6997548937797546, 0.7006739974021912, 0.7009803652763367, 0.6954656839370728, 0.6982230544090271, 0.6997548937797546, 0.6973039507865906, 0.7071078419685364, 0.6976103186607361, 0.7006739974021912, 0.6969975233078003, 0.701286792755127, 0.7015931606292725, 0.7003676295280457, 0.7022058963775635, 0.7055760025978088, 0.6914828419685364, 0.7009803652763367, 0.6957720518112183, 0.6988357901573181, 0.6939338445663452, 0.6988357901573181]


y005 = [0.6044730544090271, 0.6516544222831726, 0.6709558963775635, 0.6734068393707275, 0.673100471496582, 0.673100471496582, 0.6767769455909729, 0.6669730544090271, 0.686887264251709, 0.6743260025978088, 0.6755514740943909, 0.6783088445663452, 0.6798406839370728, 0.686887264251709, 0.6838235259056091, 0.6865808963775635, 0.6835171580314636, 0.6804534196853638, 0.6859681606292725, 0.685661792755127, 0.6865808963775635, 0.6875, 0.6890318393707275, 0.6853553652763367, 0.6865808963775635, 0.6911764740943909, 0.6902573704719543, 0.6783088445663452, 0.7015931606292725, 0.689338207244873, 0.6816789507865906, 0.6939338445663452, 0.6642156839370728, 0.6838235259056091, 0.6859681606292725, 0.6979166865348816, 0.6914828419685364, 0.6871936321258545, 0.6835171580314636, 0.6813725233078003, 0.6930146813392639, 0.6936274766921997, 0.7009803652763367, 0.6816789507865906, 0.6976103186607361, 0.6908701062202454, 0.6853553652763367, 0.6973039507865906, 0.6966911554336548, 0.6951593160629272, 0.6920955777168274, 0.7009803652763367, 0.7015931606292725, 0.6991421580314636, 0.6776960492134094, 0.6954656839370728, 0.6985294222831726, 0.688725471496582, 0.6908701062202454, 0.6951593160629272, 0.6930146813392639, 0.6930146813392639, 0.6963847875595093, 0.6927083134651184, 0.6899510025978088, 0.6966911554336548, 0.7086396813392639, 0.6963847875595093, 0.6969975233078003, 0.6954656839370728, 0.6939338445663452, 0.6991421580314636, 0.6939338445663452, 0.6979166865348816, 0.7000612616539001, 0.6948529481887817, 0.6871936321258545, 0.703125, 0.7009803652763367, 0.6979166865348816, 0.6896446347236633, 0.6905637383460999, 0.6963847875595093, 0.6960784196853638, 0.6969975233078003, 0.6979166865348816, 0.7006739974021912, 0.6945465803146362, 0.6813725233078003, 0.6976103186607361, 0.703737735748291, 0.6948529481887817, 0.6969975233078003, 0.6920955777168274, 0.6966911554336548, 0.6979166865348816, 0.6976103186607361, 0.6988357901573181, 0.702512264251709, 0.6994485259056091]


y007 = [0.5045955777168274, 0.4840686321258545, 0.501225471496582, 0.5049019455909729, 0.49816176295280457, 0.4785539209842682, 0.501225471496582, 0.5061274766921997, 0.4987744987010956, 0.4908088147640228, 0.499387264251709, 0.49693626165390015, 0.4957107901573181, 0.4950980246067047, 0.5036764740943909, 0.5036764740943909, 0.49693626165390015, 0.49632352590560913, 0.5042892098426819, 0.4938725531101227, 0.49754902720451355, 0.49142158031463623, 0.48651960492134094, 0.4791666567325592, 0.49816176295280457, 0.4852941036224365, 0.49816176295280457, 0.500612735748291, 0.5042892098426819, 0.5030637383460999, 0.5091911554336548, 0.501225471496582, 0.501225471496582, 0.4908088147640228, 0.49693626165390015, 0.5049019455909729, 0.4944852888584137, 0.5024510025978088, 0.4895833432674408, 0.5049019455909729, 0.4950980246067047, 0.5116421580314636, 0.49325981736183167, 0.49142158031463623, 0.5, 0.49693626165390015, 0.47610294818878174, 0.49693626165390015, 0.49754902720451355, 0.5049019455909729, 0.49325981736183167, 0.5024510025978088, 0.4908088147640228, 0.5067402124404907, 0.49203431606292725, 0.49693626165390015, 0.48774510622024536, 0.49142158031463623, 0.49816176295280457, 0.5049019455909729, 0.4895833432674408, 0.501838207244873, 0.4895833432674408, 0.4987744987010956, 0.49754902720451355, 0.4944852888584137, 0.5030637383460999, 0.5061274766921997, 0.4987744987010956, 0.500612735748291, 0.5055146813392639, 0.5153186321258545, 0.49203431606292725, 0.4852941036224365, 0.500612735748291, 0.5147058963775635, 0.49693626165390015, 0.49816176295280457, 0.4908088147640228, 0.5030637383460999, 0.5042892098426819, 0.49203431606292725, 0.4944852888584137, 0.4957107901573181, 0.49693626165390015, 0.5, 0.4901960790157318, 0.4944852888584137, 0.499387264251709, 0.5049019455909729, 0.48223039507865906, 0.49816176295280457, 0.500612735748291, 0.5098039507865906, 0.5085784196853638, 0.4895833432674408, 0.5049019455909729, 0.4938725531101227, 0.49754902720451355, 0.4957107901573181]
for i in range(len(y001)) :
    x1.append(i)


plt.grid(True)
plt.xlabel("Number of epochs")
plt.ylabel("")
plt.plot(x1, y001, color = 'b')
plt.plot(x1, y005, color = 'r')
plt.plot(x1, y007, color = 'g')





plt.show()