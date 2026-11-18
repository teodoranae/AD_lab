import scipy
from matplotlib import pyplot as plt
from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.data import generate_data
from pyod.models import ocsvm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# 1
# part 1

X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test= 200, n_features=3, contamination = 0.15, random_state=42)

# part 2
model_ocsvm = ocsvm.OCSVM(kernel='linear', contamination=0.15)
model_ocsvm.fit(X_train)
y_pred = model_ocsvm.predict(X_test)
y_scores = model_ocsvm.decision_function(X_test)
ba_ocsvm = balanced_accuracy_score(y_test, y_pred)
roc_auc_ocsvm = roc_auc_score(y_test, y_scores)
print(f'OCSVM: BA - {ba_ocsvm}, ROC AUC - {roc_auc_ocsvm}')

# part 3
fig = plt.figure(figsize=(12,11))
ax0 = fig.add_subplot(221, projection='3d')
ax0.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax0.set_title('Training data - Ground truth')

ax1 = fig.add_subplot(222, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax1.set_title('Test data - Ground truth')

ax2 = fig.add_subplot(223, projection='3d')
ax2.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=model_ocsvm.predict(X_train), cmap='coolwarm')
ax2.set_title('Training data - Predictions')

ax3 = fig.add_subplot(224, projection='3d')
ax3.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_pred, cmap='coolwarm')
ax3.set_title('Test data - Predictions')


plt.savefig('imgs/ex1-part3.pdf', format = 'pdf')
#plt.show()
plt.close()
# part4
model_ocsvm_rbf = ocsvm.OCSVM(kernel='rbf', contamination=0.15)
model_ocsvm_rbf.fit(X_train)
y_pred_rbf = model_ocsvm_rbf.predict(X_test)
y_scores_rbf = model_ocsvm_rbf.decision_function(X_test)
ba_ocsvm_rbf = balanced_accuracy_score(y_test, y_pred_rbf)
roc_auc_ocsvm_rbf = roc_auc_score(y_test, y_scores_rbf)
print(f'OCSVM RBF kernel: BA - {ba_ocsvm_rbf}, ROC AUC - {roc_auc_ocsvm_rbf}')


fig = plt.figure(figsize=(12,11))
ax0 = fig.add_subplot(221, projection='3d')
ax0.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax0.set_title('Training data - Ground truth')

ax1 = fig.add_subplot(222, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax1.set_title('Test data - Ground truth')

ax2 = fig.add_subplot(223, projection='3d')
ax2.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=model_ocsvm_rbf.predict(X_train), cmap='coolwarm')
ax2.set_title('Training data - Predictions')

ax3 = fig.add_subplot(224, projection='3d')
ax3.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_pred_rbf, cmap='coolwarm')
ax3.set_title('Test data - Predictions')

plt.savefig('imgs/ex1-part4-rbf.pdf', format = 'pdf')
# plt.show()
plt.close()
# part 5

model_dsvdd = DeepSVDD(epochs=20, n_features=3, contamination=0.15)
model_dsvdd.fit(X_train)
y_pred_dsvdd = model_dsvdd.predict(X_test)
y_scores_dsvdd = model_dsvdd.decision_function(X_test)
ba_dsvdd = balanced_accuracy_score(y_test, y_pred_dsvdd)
roc_auc_dsvdd = roc_auc_score(y_test, y_scores_dsvdd)
print(f'Deep SVDD: BA - {ba_dsvdd}, ROC AUC - {roc_auc_dsvdd}')

fig = plt.figure(figsize=(12,11))
ax0 = fig.add_subplot(221, projection='3d')
ax0.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax0.set_title('Training data - Ground truth')


ax1 = fig.add_subplot(222, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax1.set_title('Test data - Ground truth')

ax2 = fig.add_subplot(223, projection='3d')
ax2.scatter(X_train[:,0], X_train[:,1], X_train[:, 2], c=model_dsvdd.predict(X_train), cmap='coolwarm')
ax2.set_title('Training data - Predictions')

ax3 = fig.add_subplot(224, projection='3d')
ax3.scatter(X_test[:,0], X_test[:,1], X_test[:, 2], c=y_pred_dsvdd, cmap='coolwarm')
ax3.set_title('Test data - Predictions')

plt.savefig('imgs/ex1-part5.pdf', format = 'pdf')
plt.show()
plt.close()

# 2

# part 1
data = scipy.io.loadmat('data/cardio.mat')
X= data['X']
y= data['y'].ravel()
y = (-2 * y + 1).astype(int)     # part 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# part 2
grid_params = {'ocsvm__kernel': ['linear', 'rbf'], 'ocsvm__nu': [0.05, 0.1, 0.3], 'ocsvm__gamma': [0.01, 0.1, 1]}
# part 4
pipeline = Pipeline([('scaler', StandardScaler()), ('ocsvm', OneClassSVM())])

# parts 3
grid = GridSearchCV(estimator=pipeline, param_grid=grid_params, scoring='balanced_accuracy', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
#part 6
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)
print('Best parameters:', grid.best_params_)
test_ba = balanced_accuracy_score(y_test, y_test_pred)
print('Balanced accuracy:', test_ba)


# 3
# part 1
data= scipy.io.loadmat('data/shuttle.mat')
X= data['X']
y= data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# part 2
model_ocsvm = ocsvm.OCSVM()
model_ocsvm.fit(X_train_scaled)
y_pred_ocsvm = model_ocsvm.predict(X_test_scaled)
y_scores_ocsvm = model_ocsvm.decision_function(X_test_scaled)
ba_ocsvm = balanced_accuracy_score(y_test, y_pred_ocsvm)
roc_auc_ocsvm = roc_auc_score(y_test, y_scores_ocsvm)

print(f'OCSVM: BA - {ba_ocsvm}, ROC AUC - {roc_auc_ocsvm} ')

model_dsvdd = DeepSVDD(n_features=X_train_scaled.shape[1], epochs=20)
model_dsvdd.fit(X_train_scaled)
y_pred_dsvdd = model_dsvdd.predict(X_test_scaled)
y_scores_dsvdd = model_dsvdd.decision_function(X_test_scaled)
ba_dsvdd = balanced_accuracy_score(y_test, y_pred_dsvdd)
roc_auc_dsvdd = roc_auc_score(y_test, y_scores_dsvdd)

print(f'D-SVDD: BA - {ba_dsvdd}, ROC AUC - {roc_auc_dsvdd} ')

# part 3
archs = [[16, 8], [32, 16, 8], [64, 32, 16], [128, 64, 32, 16]]
for arch in archs:
    model = DeepSVDD(hidden_neurons=arch, n_features=X_train_scaled.shape[1], epochs=20)
    model.fit(X_train_scaled)
    y_pred_arch = model.predict(X_test_scaled)
    y_scores_arch = model.decision_function(X_test_scaled)
    ba_arch = balanced_accuracy_score(y_test, y_pred_arch)
    roc_auc_arch = roc_auc_score(y_test, y_scores_arch)
    print(f'Architecture {str(arch)}: BA {ba_arch}, ROC AUC {roc_auc_arch}')
