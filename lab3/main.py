# 1
import scipy
from matplotlib import pyplot as plt
from pyod.models.devnet import train_and_test
from pyod.models.dif import DIF
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# # part 1
dataset, _ = make_blobs(n_samples=500, n_features=2, random_state=42)

# part 2
proj_vectors = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=5)
proj_vectors /= np.linalg.norm(proj_vectors, axis=1, keepdims=True)
test_dataset = np.random.uniform(low=-3, high=3, size=(500, 2))  # for part 3
def anomaly_scores(n_bins):
    probs = np.zeros((dataset.shape[0], 5))
    histograms = []
    for i in range(5):
        projected = np.dot(dataset, proj_vectors[i])
        hist_range = (projected.min() - 1, projected.max() + 1)
        counts, bin_edges = np.histogram(projected, bins=n_bins, range=hist_range)
        bin_probs = counts / len(projected)
        histograms.append((bin_probs, bin_edges))
        bin_indices = np.digitize(projected, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        probs[:, i] = bin_probs[bin_indices]
    # part 2 - anomaly scores
    train_scores = np.mean(probs, axis=1)

    # part 3
    y_test = np.dot(test_dataset, proj_vectors.T)
    test_probs = np.zeros_like(y_test)
    for i in range(5):
            bin_probs, bin_edges = histograms[i]
            bin_indices = np.digitize(y_test[:, i], bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            test_probs[:, i] = bin_probs[bin_indices]
    test_scores = test_probs.mean(axis=1)
    return train_scores, test_scores

# part 2
train_scores, test_scores = anomaly_scores(20)
# print(train_scores)

# part 3 plot
plt.figure(figsize=(8, 6))
plt.scatter(test_dataset[:, 0], test_dataset[:, 1], c=test_scores, cmap='viridis')
plt.colorbar(label = 'Anomaly Score')
plt.title('Test data points and anomaly scores')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()
plt.savefig('imgs/ex1-part3.pdf', format='pdf')
plt.close()

# part 4
bins = [5, 10, 20, 30]
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for ax, no_bin in zip(axes.flat, bins):
    _, test_scores = anomaly_scores(no_bin)
    ax.scatter(test_dataset[:, 0], test_dataset[:, 1], c=test_scores, cmap='viridis')
    ax.set_title(f'Bins = {no_bin}')
plt.suptitle('Test data points and anomaly scores across different bin numbers')
# plt.show()
plt.savefig('imgs/ex2-part4.pdf', format='pdf')
plt.close()

# 2
# part 1
dataset2, _ =make_blobs(n_samples=[500, 500], n_features=2, centers = [( 10, 0), ( 0, 10)], cluster_std=1, random_state=42)

# parts 2
model_if = IForest(contamination=0.02)
model_if.fit(dataset2)
test_dataset2 = np.random.uniform(low=-10, high=20, size=(1000, 2))
# part 3
test_scores2 = model_if.decision_function(test_dataset2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc0= axes[0].scatter(test_dataset2[:, 0], test_dataset2[:, 1], c=test_scores2, cmap='viridis')
axes[0].set_title('Anomaly scores from isolation forest')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
fig.colorbar(sc0, ax = axes[0], label = 'Anomaly Score')
# plt.show()

# part 4
model_dif = DIF(contamination=0.02)
model_dif.fit(dataset2)
test_scores2_dif = model_dif.decision_function(test_dataset2)
sc1 =axes[1].scatter(test_dataset2[:, 0], test_dataset2[:, 1], c=test_scores2_dif, cmap='viridis')
axes[1].set_title('Anomaly scores from deep isolation forest')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
fig.colorbar(sc1, ax=axes[1], label = 'Anomaly Score')
# plt.show()

model_loda = LODA(contamination=0.02)
model_loda.fit(dataset2)
test_scores2_loda = model_loda.decision_function(test_dataset2)
sc2 =axes[2].scatter(test_dataset2[:, 0], test_dataset2[:, 1], c=test_scores2_loda, cmap='viridis')
axes[2].set_title('Anomaly scores from LODA')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
fig.colorbar(sc2, ax=axes[2], label = 'Anomaly Score')
# plt.show()
plt.tight_layout()
plt.savefig('imgs/ex2-comparison.pdf', format='pdf')
plt.close()

# part 5
hidden_neurons =[[16, 8], [32, 16], [64, 32]]
fig, axes = plt.subplots(2, 3, figsize=(18, 5))
bins = [5, 10, 30]
for ax, neurons in zip(axes[0], hidden_neurons):
    model_dif = DIF(hidden_neurons=neurons, contamination=0.02)
    model_dif.fit(dataset2)
    scores_dif = model_dif.decision_function(test_dataset2)
    sc = ax.scatter(test_dataset2[:, 0], test_dataset2[:, 1], c=scores_dif, cmap = 'viridis')
    ax.set_title(f'DIF with {neurons} hidden neurons')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.colorbar(sc, ax = ax, label = 'Anomaly Score')

for ax, n_bins in zip(axes[1], bins):
    model_loda = LODA(n_bins=n_bins, contamination=0.02)
    model_loda.fit(dataset2)
    scores_loda = model_loda.decision_function(test_dataset2)
    sc = ax.scatter(test_dataset2[:, 0], test_dataset2[:, 1], c=scores_loda, cmap = 'viridis')
    ax.set_title(f'LODA with {n_bins} bins')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.colorbar(sc, ax = ax, label = 'Anomaly Score')

plt.tight_layout()
plt.savefig("imgs/ex2-part5.pdf", format="pdf", bbox_inches="tight")
# plt.show()
plt.close()


# part 6- 3d, mostly code duplication
dataset3, _ =make_blobs(n_samples=[500, 500], n_features=2, centers = [(0, 10, 0), (10, 0, 10)], cluster_std=1, random_state=42)

model_if3d = IForest(contamination=0.02)
model_if3d.fit(dataset3)
test_dataset3 = np.random.uniform(low=-10, high=20, size=(1000, 3))

test_scores_if3d = model_if3d.decision_function(test_dataset3)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})

sc0= axes[0].scatter(test_dataset3[:, 0], test_dataset3[:, 1], test_dataset3[:, 2], c=test_scores_if3d, cmap='viridis')
axes[0].set_title('Anomaly scores from isolation forest')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_zlabel('x3')
fig.colorbar(sc0, ax = axes[0], label = 'Anomaly Score')
# plt.show()

# part 4
model_dif3d = DIF(contamination=0.02)
model_dif3d.fit(dataset3)
test_scores_dif3d = model_dif3d.decision_function(test_dataset3)
sc1 =axes[1].scatter(test_dataset3[:, 0], test_dataset3[:, 1], test_dataset3[:, 2],c=test_scores_dif3d, cmap='viridis')
axes[1].set_title('Anomaly scores from deep isolation forest')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_zlabel('x3')

fig.colorbar(sc1, ax=axes[1], label = 'Anomaly Score')
# plt.show()

model_loda3d = LODA(contamination=0.02)
model_loda3d.fit(dataset3)
test_scores_loda3d = model_loda3d.decision_function(test_dataset3)
sc2 =axes[2].scatter(test_dataset3[:, 0], test_dataset3[:, 1], test_dataset3[:, 2], c=test_scores_loda3d, cmap='viridis')
axes[2].set_title('Anomaly scores from LODA')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_zlabel('x3')
fig.colorbar(sc2, ax=axes[2], label = 'Anomaly Score')
# plt.show()
plt.tight_layout()
plt.savefig('imgs/ex2-comparison3d.pdf', format='pdf')
plt.close()

# part 5
hidden_neurons =[[16, 8], [32, 16], [64, 32]]
fig, axes = plt.subplots(2, 3, figsize=(18, 5), subplot_kw={"projection": "3d"})
bins = [5, 10, 30]
for ax, neurons in zip(axes[0], hidden_neurons):
    model_dif = DIF(hidden_neurons=neurons, contamination=0.02)
    model_dif.fit(dataset3)
    scores_dif = model_dif.decision_function(test_dataset3)
    sc = ax.scatter(test_dataset3[:, 0], test_dataset3[:, 1], test_dataset3[:, 2],c=scores_dif, cmap = 'viridis')
    ax.set_title(f'DIF with {neurons} hidden neurons')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    fig.colorbar(sc, ax = ax, label = 'Anomaly Score')

for ax, n_bins in zip(axes[1], bins):
    model_loda = LODA(n_bins=n_bins, contamination=0.02)
    model_loda.fit(dataset3)
    scores_loda = model_loda.decision_function(test_dataset3)
    sc = ax.scatter(test_dataset3[:, 0], test_dataset3[:, 1], test_dataset3[:, 2], c=scores_loda, cmap = 'viridis')
    ax.set_title(f'LODA with {n_bins} bins')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    fig.colorbar(sc, ax = ax, label = 'Anomaly Score')

plt.tight_layout()
plt.savefig("imgs/ex2-different-configs-3d.pdf", format="pdf", bbox_inches="tight")
# plt.show()
plt.close()

# 3
# part 1
data = scipy.io.loadmat('data/shuttle.mat')
X = data['X']
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# part 2
model_iforest = IForest()
model_iforest.fit(X_train)
y_pred_iforest = model_iforest.predict(X_test)

model_loda = LODA()
model_loda.fit(X_train)
y_pred_loda = model_loda.predict(X_test)

model_diforest = DIF()
model_diforest.fit(X_train)
y_pred_diforest = model_diforest.predict(X_test)
y_scores_iforest = model_iforest.decision_function(X_test)
y_scores_loda = model_loda.decision_function(X_test)
y_scores_diforest = model_diforest.decision_function(X_test)

ba_if = balanced_accuracy_score(y_test, y_pred_iforest )
ba_dif = balanced_accuracy_score(y_test, y_pred_diforest )
ba_loda = balanced_accuracy_score(y_test, y_pred_loda )
print(f'BA : IF - {ba_if}, DIF - {ba_dif}, LODA - {ba_loda}')
roc_auc_if = roc_auc_score(y_test, y_scores_iforest)
roc_auc_dif = roc_auc_score(y_test, y_scores_diforest)
roc_auc_loda = roc_auc_score(y_test, y_scores_loda)
print(f'ROC AUC: IF - {roc_auc_if}, DIF - {roc_auc_dif}, LODA - {roc_auc_loda}')

# for the 10 train-test splits
ba_if_list, ba_dif_list, ba_loda_list = [], [], []
roc_if_list, roc_dif_list, roc_loda_list = [], [], []

for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_iforest = IForest()
    model_iforest.fit(X_train)
    y_pred_iforest = model_iforest.predict(X_test)

    model_loda = LODA()
    model_loda.fit(X_train)
    y_pred_loda = model_loda.predict(X_test)

    model_diforest = DIF()
    model_diforest.fit(X_train)
    y_pred_diforest = model_diforest.predict(X_test)

    y_scores_iforest = model_iforest.decision_function(X_test)
    y_scores_loda = model_loda.decision_function(X_test)
    y_scores_diforest = model_diforest.decision_function(X_test)

    ba_if = balanced_accuracy_score(y_test, y_pred_iforest)
    ba_if_list.append(ba_if)
    ba_dif = balanced_accuracy_score(y_test, y_pred_diforest)
    ba_dif_list.append(ba_dif)
    ba_loda = balanced_accuracy_score(y_test, y_pred_loda)
    ba_loda_list.append(ba_loda)
    roc_auc_if = roc_auc_score(y_test, y_scores_iforest)
    roc_if_list.append(roc_auc_if)
    roc_auc_dif = roc_auc_score(y_test, y_scores_diforest)
    roc_dif_list.append(roc_auc_dif)
    roc_auc_loda = roc_auc_score(y_test, y_scores_loda)
    roc_loda_list.append(roc_auc_loda)

print(f'Mean BA: IF - {np.mean(ba_if_list)}, DIF - {np.mean(ba_dif_list)}, LODA - {np.mean(ba_loda_list)}')
print(f'Mean ROC AUC: IF -{np.mean(roc_if_list)}, DIF - {np.mean(roc_dif_list)}, LODA - {np.mean(roc_loda_list)}')