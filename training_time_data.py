import joblib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# load the configurations
conf = joblib.load('models/results')

conf_df = pd.DataFrame(conf)[0:36]
conf = {}
for c in conf_df.columns:
    conf[c] = conf_df[c].values.tolist()

times = pd.DataFrame()

times['ZZFeatureMap1'] = conf_df.iloc[0:12]['training_time(s)'].values
times['ZZFeatureMap3'] = conf_df.iloc[12:24]['training_time(s)'].values
times['PauliFeatureMap'] = conf_df.iloc[24:]['training_time(s)'].values

idx_ansatz = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27]
times['RealAmplitudes'] = conf_df.iloc[idx_ansatz]['training_time(s)'].values
idx_ansatz = [i+4 for i in idx_ansatz]
times['TwoLocal'] = conf_df.iloc[idx_ansatz]['training_time(s)'].values
idx_ansatz = [i+4 for i in idx_ansatz]
times['EfficientSU2'] = conf_df.iloc[idx_ansatz]['training_time(s)'].values

idx_opt = [*range(0, 36, 4)]
times['SPSA'] = np.concatenate((conf_df.iloc[idx_opt]['training_time(s)'].values,
                                np.array([np.nan, np.nan, np.nan])))
idx_opt = [i+1 for i in idx_opt]
times['QN-SPSA'] = np.concatenate((conf_df.iloc[idx_opt]['training_time(s)'].values,
                                   np.array([np.nan, np.nan, np.nan])))
idx_opt = [i+1 for i in idx_opt]
times['COBYLA'] = np.concatenate((conf_df.iloc[idx_opt]['training_time(s)'].values,
                                  np.array([np.nan, np.nan, np.nan])))
idx_opt = [i+1 for i in idx_opt]
times['Nelder-Mead'] = np.concatenate((conf_df.iloc[idx_opt]['training_time(s)'].values,
                                       np.array([np.nan, np.nan, np.nan])))

sns.boxplot(times, orient='horizontal', color='tab:blue', flierprops={"marker": "x"})
plt.xlabel('Training time (s)')
plt.title('Training time statistics')
plt.show()
