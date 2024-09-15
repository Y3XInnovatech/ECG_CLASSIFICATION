
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import wfdb
import ast
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path,f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path,f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path='C:/TheCave/work-stuff/ECG/Datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(os.path.join(path,'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
# %%
print("loading data")
X = load_raw_data(Y, sampling_rate, path)

# %%
agg_df = pd.read_csv(os.path.join(path,'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


# %%
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# %%
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# %%
test_fold = 10

X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass



mlb = MultiLabelBinarizer()
y_train_encoded = pd.DataFrame(mlb.fit_transform(y_train), columns=mlb.classes_)




# %%
standard_scaler=StandardScaler()
num_sample,time_stamps,lead_ecg=X_train.shape

# %%
X_train_new=X_train.reshape(num_sample,-1)
standard_scaler.fit_transform(X_train_new)

# %%
# X_train=X_train_new.reshape(num_sample,time_stamps,lead_ecg)

# %%
# from sklearn.decomposition import PCA
# pca=PCA(n_components=3)
# X_pca=pca.fit_transform(X_train_new)


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# for class_label in range(y_train_encoded.shape[1]):
#     indices = np.where(np.asarray(y_train_encoded)[:, class_label] == 1)
#     marker = 'o' if class_label == 0 else '^'  # Use different markers for each class
#     ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], marker=marker, label=f'Class {class_label}')

# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# ax.set_title('3D PCA Analysis ')
# ax.legend()

# plt.show()



print("tnse started")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_train_new)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for class_label in range(y_train_encoded.shape[1]):
    indices = np.where(np.asarray(y_train_encoded)[:, class_label] == 1)
    marker = 'o' if class_label == 0 else '^'
    ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2], marker=marker, label=f'Class {class_label}')

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('3D t-SNE Analysis')
ax.legend()
plt.show()
