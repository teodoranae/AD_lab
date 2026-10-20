from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc, balanced_accuracy_score
import numpy as np

# ex 1
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, random_state=42, contamination=0.1)
inliers = X_train[y_train==0]
outliers = X_train[y_train==1]
plt.scatter(inliers[:, 0], inliers[:, 1], color='g', label='Inliers')
plt.scatter(outliers[:,0], outliers[:,1], color= 'r', label='Outliers')
plt.title('Inliers and Outliers')
plt.legend()
plt.savefig('fig/ex1.pdf', format='pdf')
# plt.show()
plt.close()

# ex2
print('ex 2:')
KNN_model = KNN(contamination=0.1)
KNN_model.fit(X_train)
y_train_pred = KNN_model.predict(X_train)
y_test_pred = KNN_model.predict(X_test)
TN1, FP1, FN1, TP1  = confusion_matrix(y_train, y_train_pred).ravel()
print(f'TP={TP1}, FN={FN1}, FP={FP1}, TN={TN1} (train)')
print(f'Balanced accuracy (train): {0.5 * (TP1/(TP1+FN1) + TN1/(FP1+TN1))}')

TN2, FP2, FN2, TP2  = confusion_matrix(y_test, y_test_pred).ravel()
print(f'TP={TP2}, FN={FN2}, FP={FP2}, TN={TN2} (test)')
print(f'Balanced accuracy (test): {0.5 * (TP2/(TP2+FN2) + TN2/(FP2+TN2))}')

scores_test = KNN_model.decision_function(X_test)
FPR, TPR,thresh = roc_curve(y_test, scores_test)
roc_auc= auc(FPR, TPR)

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('FP Rate')
plt.ylabel('Tp Rate')
plt.title('Contamination = 0.1')
plt.legend()
plt.grid(True)
plt.savefig('fig/ex2_first_contamination.pdf', format='pdf')
#plt.show()
plt.close()


#for contamination 0.3
print("contamination = 0.3:")
KNN_model = KNN(contamination=0.3)
KNN_model.fit(X_train)
y_test_pred = KNN_model.predict(X_test)
y_train_pred = KNN_model.predict(X_train)
TN1, FP1, FN1, TP1  = confusion_matrix(y_train, y_train_pred).ravel()
print(f'TP={TP1}, FN={FN1}, FP={FP1}, TN={TN1} (train)')
print(f'Balanced accuracy (train): {0.5 * (TP1/(TP1+FN1) + TN1/(FP1+TN1))}')

TN2, FP2, FN2, TP2  = confusion_matrix(y_test, y_test_pred).ravel()
print(f'TP={TP2}, FN={FN2}, FP={FP2}, TN={TN2} (test)')
print(f'Balanced accuracy (test): {0.5 * (TP2/(TP2+FN2) + TN2/(FP2+TN2))}')

scores_test = KNN_model.decision_function(X_test)
FPR, TPR,_ = roc_curve(y_test, scores_test)
roc_auc= auc(FPR, TPR)

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('FP Rate')
plt.ylabel('Tp Rate')
plt.title('Contamination = 0.3')
plt.legend()
plt.grid(True)
plt.savefig('fig/ex2_second_contamination.pdf', format='pdf')
plt.show()
plt.close()


# ex 3
print('\nex 3:')
X_uni_train, _, y_uni_train, _ = generate_data(n_train=1000, contamination=0.1, random_state=42, n_features = 1)
mean = np.mean(X_uni_train)
std = np.std(X_uni_train)
def z_score(x):
    return np.abs((x - mean) / std)

thresh = np.quantile([z_score(x) for x in X_uni_train], 0.9)
y_pred = [1 if z_score(x) > thresh else 0 for x in X_uni_train]

TN, FP, FN, TP = confusion_matrix(y_uni_train, y_pred).ravel()
print(f'TN={TN}, FP={FP}, FN={FN}, TP={TP}')
print('Balanced accuracy: ' + str(0.5 * (TP/(TP+FP))) + '\n')

# ex 4
print('\nex 4:')
n_samples = 1000
n_features = 3
contamination = 0.1
mu = np.array([2.0, 5.0, -1.0])
sigma = np.array([
    [1.0, 0.8, 0.5],
    [0.8, 2.0, 0.3],
    [0.5, 0.3, 1.5]
])
L=np.linalg.cholesky(sigma)
n_outliers = int(n_samples * contamination)
n_inliers = int(n_samples - n_outliers)
x_in = np.random.randn(n_inliers, n_features)
y_in = x_in @ L.T + mu

x_out = np.random.randn(n_outliers, n_features)
y_out = (x_out @ L.T) + mu + np.array([10, 10, 10])

Y = np.vstack((y_in, y_out))
y_true = np.array([0] * n_inliers + [1] * n_outliers)

mean_y = np.mean(Y, axis = 0)
std_y = np.std(Y, axis = 0)
z_scores = np.abs((Y-mean_y) / std_y)
z = np.linalg.norm(z_scores, axis=1)

thresh = np.quantile(np.abs(z), 0.9)
y_pred = (z>thresh).astype(int)
accuracy = balanced_accuracy_score(y_true, y_pred)
confusion_matrix = confusion_matrix(y_true, y_pred)

print(f'Threshold: {thresh:.2f}')
print(f'Balanced accuracy: {accuracy:.2f}%')
print('Confusion matrix:\n', confusion_matrix)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[y_true==0, 0], Y[y_true ==0, 1], Y[y_true == 0,2], c ='b', s=20, label = 'True inliers')
ax.scatter(Y[y_true==1, 0], Y[y_true ==1, 1], Y[y_true == 1,2], c ='r', s=30, label = 'True outliers')
ax.set_title('Z-score anomaly detection in a multidimensional dataset')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.savefig('fig/ex4.pdf', format='pdf')

plt.show()