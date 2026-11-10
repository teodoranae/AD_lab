from pyod.models.combination import maximization, average
from pyod.models.lof import LOF
from pyod.utils import standardizer
from sklearn.datasets import make_blobs
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#ex 1
def create_data_1d(mean_eps, std_eps, mean_x, std_x, num_samples=1000):
    a = 2
    b = 1
    x = np.random.normal(mean_x, std_x, num_samples)
    eps = np.random.normal(mean_eps, std_eps, num_samples)
    y = a * x + b + eps
    X = np.column_stack([x, np.ones_like(x)])
    U, S, V = np.linalg.svd(X, full_matrices=False)
    H = U  @ U.T
    leverage_scores= np.diag(H)
    threshold = np.quantile(leverage_scores, 0.9)
    high_leverage = np.where(leverage_scores > threshold)[0]
    low_leverage = np.where(leverage_scores <= threshold)[0]
    return x, y, leverage_scores, high_leverage, low_leverage

fig, axes = plt.subplots(2, 2, figsize =(10, 8))
axes = axes.flatten()
options=[{'title': 'Low variance, close to model', 'std_x': 1, 'std_eps': 0.5},
    {'title': 'High variance on x', 'std_x': 8, 'std_eps': 0.5},
    {'title': 'High variance on y', 'std_x': 1, 'std_eps': 3.5},
    {'title': 'High variance on both x and y', 'std_x': 8, 'std_eps': 8}
]
for i, option in enumerate(options):
    ax = axes[i]
    x, y, leverage_scores, high_leverage, low_leverage = create_data_1d(0, option['std_eps'], 0, option['std_x'])
    ax.scatter(x[low_leverage], y[low_leverage], color='b', label='Low leverage')
    ax.scatter(x[high_leverage], y[high_leverage], color='r', label='High leverage')
    ax.set_title(option['title'])
    ax.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
# plt.show()
# plt.savefig('imgs/ex1_1d.pdf', format='pdf')
# plt.close()


def create_data_2d(mean_eps, std_eps, mean_x, std_x1, std_x2, num_samples=1000):
    a1, a2 = 2.0, 1.5
    b = 1.0
    x1 = np.random.normal(mean_x, std_x1, num_samples)
    x2 = np.random.normal(mean_x, std_x2, num_samples)
    eps = np.random.normal(mean_eps, std_eps, num_samples)
    y = a1 * x1 + a2 * x2 + b + eps
    X = np.column_stack([x1, x2, np.ones_like(x1)])
    U, S, V = np.linalg.svd(X, full_matrices=False)
    H = U @ U.T
    leverage_scores = np.diag(H)
    threshold = np.quantile(leverage_scores, 0.9)
    high_leverage = np.where(leverage_scores > threshold)[0]
    low_leverage = np.where(leverage_scores <= threshold)[0]
    return x1, x2, y, leverage_scores, high_leverage, low_leverage

fig = plt.figure(figsize=(12, 10))
options = [
    {'title': 'Low variance, close to model', 'std_x1': 1, 'std_x2': 1, 'std_eps': 0.5},
    {'title': 'High variance on x', 'std_x1': 6, 'std_x2': 6, 'std_eps': 0.5},
    {'title': 'High variance on y ', 'std_x1': 1, 'std_x2': 1, 'std_eps': 5.0},
    {'title': 'High variance on both x and y', 'std_x1': 6, 'std_x2': 6, 'std_eps': 5}
]

for i, option in enumerate(options):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    x1, x2, y, leverage_scores, high_leverage, low_leverage = create_data_2d(0,option['std_eps'],0,option['std_x1'],option['std_x2'])
    ax.scatter(x1[low_leverage], x2[low_leverage], y[low_leverage], color='b', label='Low leverage')
    ax.scatter(x1[high_leverage], x2[high_leverage], y[high_leverage], color='r', label='High leverage')
    ax.set_title(option['title'], fontsize=10)

# plt.tight_layout()
# plt.savefig('imgs/ex1_2d.pdf', format='pdf')
# plt.show()

# ex 2
X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_features=2,contamination=0.1, n_clusters=2)
for k in [5, 7, 13]:
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    knn_model = KNN(n_neighbors=k)
    knn_model.fit(X_train)
    y_pred_train = knn_model.predict(X_train)
    y_pred_test = knn_model.predict(X_test)
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    axs[0, 0].set_title(f'Ground truth labels (train)')
    axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
    axs[0, 1].set_title(f'Predicted labels (train) ')
    axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    axs[1, 0].set_title(f'Ground truth labels (test)')
    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
    axs[1, 1].set_title(f'Predicted labels (test)')
    fig.suptitle(f'KNN: k={k}')
    # plt.tight_layout()
    # plt.savefig(f'imgs/ex2_k={k}.pdf', format='pdf')
    # plt.show()
    # plt.close()

# ex 3
contamination= 0.07
clusters, y_true=make_blobs(n_samples=[200, 100] , n_features=2, centers=[(-10,-10), (10, 10)], cluster_std=[2.0, 6.0], random_state=42)

for k in [5, 13, 21]:
    knn_model = KNN(contamination=contamination, n_neighbors=k)
    knn_model.fit(clusters)
    lof_model = LOF(contamination=contamination, n_neighbors=k)
    lof_model.fit(clusters)
    knn_pred = knn_model.predict(clusters)
    lof_pred = lof_model.predict(clusters)

    fig, ax = plt.subplots(ncols = 2, nrows=1, figsize = (10,5))
    fig.suptitle(f'K = {k}')
    ax[0].scatter(clusters[:, 0], clusters[:, 1],c=knn_pred)
    ax[0].set_title("KNN")
    ax[1].scatter(clusters[:, 0], clusters[:, 1],c=lof_pred)
    ax[1].set_title("LOF")
    # plt.savefig(f'imgs/ex_3_knn_vs_lof_k={k}.pdf', format ='pdf')
    # plt.show()
    # plt.close()

# ex4
data = scipy.io.loadmat("./data/cardio.mat")
X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

contamination = 0.05
k_vals = np.linspace(30, 120, 10, dtype=int)
train_scores_knn, test_scores_knn = [], []
train_scores_lof, test_scores_lof = [], []
for k in k_vals:
    knn_model = KNN(n_neighbors=k, contamination=contamination)
    knn_model.fit(X_train_norm)
    train_score_knn = knn_model.decision_scores_
    test_score_knn = knn_model.decision_function(X_test_norm)
    thresh_knn = np.quantile(train_score_knn, 1 - contamination)
    ypred_train_knn = (train_score_knn >= thresh_knn).astype(int)
    ypred_test_knn = (test_score_knn >= thresh_knn).astype(int)
    balanced_accuracy_train_knn = balanced_accuracy_score(y_train, ypred_train_knn)
    balanced_accuracy_test_knn = balanced_accuracy_score(y_test, ypred_test_knn)
    train_scores_knn.append(train_score_knn)
    test_scores_knn.append(test_score_knn)

    lof_model = LOF(contamination=contamination, n_neighbors=k)
    lof_model.fit(X_train_norm)
    train_score_lof = lof_model.decision_scores_
    test_score_lof = lof_model.decision_function(X_test_norm)
    thresh_lof = np.quantile(train_score_lof, 1 - contamination)
    ypred_train_lof = (train_score_lof >= thresh_lof).astype(int)
    ypred_test_lof = (test_score_lof >= thresh_lof).astype(int)
    balanced_accuracy_train_lof = balanced_accuracy_score(y_train, ypred_train_lof)
    balanced_accuracy_test_lof = balanced_accuracy_score(y_test, ypred_test_lof)
    train_scores_lof.append(train_score_lof)
    test_scores_lof.append(test_score_lof)
    print(f'k={k} | KNN BA train = {balanced_accuracy_train_knn: .2f} ; test = {balanced_accuracy_test_knn} | LOF BA train = {balanced_accuracy_train_lof:.2f} ; test = {balanced_accuracy_test_lof:.2f}')

train_scores_knn = np.vstack(train_scores_knn)
test_scores_knn = np.vstack(test_scores_knn)
train_scores_lof = np.vstack(train_scores_lof)
test_scores_lof = np.vstack(test_scores_lof)

def eval_ensemble(model, train_scores, test_scores):
    train_std, test_std = standardizer(train_scores.T, test_scores.T)
    for strategy, combiner in [('Average', average), ('Maximization', maximization)]:
        train_comb = combiner(train_std)
        test_comb = combiner(test_std)
        thresh = np.quantile(train_comb, 1 - contamination)
        ypred_train = (train_comb >= thresh).astype(int)
        ypred_test = (test_comb >= thresh).astype(int)
        balanced_accuracy_train = balanced_accuracy_score(y_train, ypred_train)
        balanced_accuracy_test = balanced_accuracy_score(y_test, ypred_test)
        print(f'{model} | {strategy} BA train = {balanced_accuracy_train:.2f} ; test = {balanced_accuracy_test:.2f}')

eval_ensemble('KNN', train_scores_knn, test_scores_knn)
eval_ensemble('LOF', train_scores_lof, test_scores_lof)