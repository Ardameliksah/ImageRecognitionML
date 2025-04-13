import numpy as np
import matplotlib.pyplot as plt
train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')

print(train_images.shape) # (20000, 28, 28)
print(test_images.shape) # (5000, 28, 28)
print(train_images[0].shape) # 28 x 28
print(train_labels) 
##Feature Extraction

train_flat = train_images.reshape(train_images.shape[0], -1)
test_flat = test_images.reshape(test_images.shape[0], -1)
train_normal = train_flat/255
test_normal = test_flat/255

# PCA
#https://medium.com/technological-singularity/build-a-principal-component-analysis-pca-algorithm-from-scratch-7515595bf08b


def PCA_from_Scratch(X, n_components):
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-4)
    cov_mat = np.cov(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)  # Use eigh for symmetric matrices (covariance is symmetric)
    sort_indices = np.argsort(eigen_values)[::-1]
    principal_components = eigen_vectors[:, sort_indices[:n_components]]

    return principal_components


def transform(X, principal_components):
    X = X.copy()
    return X.dot(principal_components)



# LDA 
#https://www.kaggle.com/code/egazakharenko/linear-discriminant-analysis-lda-from-scratch
class LDA():
  def __init__(self, n_components=None):
     self.n_components = n_components
  def fit(self,X,y):
     self.X = X
     self.y = y
     samples = X.shape[0]
     features= X.shape[1]
     classes, cls_counts = np.unique(y,return_counts=True)
     priors = cls_counts/samples
     X_mean = np.array([X[y==cls].mean(axis=0) for cls in classes])
     betweenCLSdeviation = X_mean - X.mean(axis=0)
     withinCLSdeviation = X - X_mean[y]

     Sb = priors* betweenCLSdeviation.T @ betweenCLSdeviation
     Sw = withinCLSdeviation.T @ withinCLSdeviation / samples
     Sw_inv = np.linalg.pinv(Sw)
     eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
     self.dvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
     self.weights = X_mean @ self.dvecs @ self.dvecs.T
     self.bias = np.log(priors) - 0.5 * np.diag(X_mean @ self.weights.T)
     if self.n_components is None:
        self.n_components = min(classes.size - 1, features)
  def transform(self, X):
    return X @ self.dvecs[:, : self.n_components]

  def predict(self, X_test):
    scores = X_test @ self.weights.T + self.bias
    return np.argmax(scores, axis=1)


lda = LDA()
lda.fit(train_normal,train_labels)
lda_train = lda.transform(train_normal)
lda_test = lda.transform(test_normal)



def macroF1(y_pred, y_true):
    labels = np.unique(y_true)
    f1_scores = []

    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
def score(y_pred, y_true):
        accuracy = np.mean(y_pred == y_true)
        return accuracy

class myMultiClassLogisticRegression():
    def __init__(self, learning_rate=0.1, maxiterations=10000, mu=0.0):
        
        self.learning_rate = learning_rate
        self.maxiterations = maxiterations
        self.mu = mu  # regularization strength
        self.W = None  # weight matrix

    def OneHotEncoder(self, y_train, num_classes):

        y_trainNew = np.zeros((y_train.shape[0], num_classes))
        for i, label in enumerate(y_train):
            y_trainNew[i, label] = 1
        return y_trainNew

    def softmax(self, Z):
        
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # stability
        exp_Z = np.exp(Z_shifted)
        return np.real(exp_Z / np.sum(exp_Z, axis=1, keepdims=True))

    def loss(self, X, Y, W):
        
        N = X.shape[0]
        logits = X @ W  # shape (N, C)
        P = self.softmax(logits)  # shape (N, C)
        eps = 1e-15
        cross_entropy = -np.mean(np.sum(Y * np.log(P + eps), axis=1))
        reg_term = self.mu * np.sum(W**2)
        return cross_entropy + reg_term

    def gradient(self, X, Y, W):
        
        N = X.shape[0]
        logits = X @ W  # shape (N, C)
        P = self.softmax(logits)  # shape (N, C)
        grad = (1.0 / N) * (X.T @ (P - Y)) + 2.0 * self.mu * W
        return np.real(grad)

    def fit(self, X_train, y_train):
       
        self.n_images, self.n_features = X_train.shape
        num_classes = len(np.unique(y_train))
        Y_onehot = self.OneHotEncoder(y_train, num_classes)
        self.W = np.zeros((self.n_features, num_classes))
        for _ in range(self.maxiterations):
            grad = self.gradient(X_train, Y_onehot, self.W)
            self.W -= self.learning_rate * grad

    def predict(self, X_test):
       
        logits = X_test @ self.W  
        P = self.softmax(logits)  
        return np.argmax(P, axis=1)

class OneVsAllLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, _lambda=0.1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._lambda = _lambda
        self.classifiers = None
        self.classes_ = None

    def sigmoid(self, z):
        return np.real(1.0 / (1 + np.exp(-z)))

    def cost_fun(self, W, X, y):
        N  = X.shape[0]
        h = self.sigmoid(X @ W)
        eps = 1e-15
        cost = -(1.0/N) * (y.T @ np.log(h + eps) + (1 - y).T @ np.log(1 - h + eps)) + (self._lambda / (2*N)) * np.sum(W**2)
        return cost

    def gradient(self, W, X, y):
        N = X.shape[0]
        h = self.sigmoid(X @ W)
        grad = (1.0/N) * X.T @ (h - y) + (self._lambda/N) * W
        return np.real(grad)

    def fit(self, X, y):
        m, n = X.shape
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        all_theta = np.zeros((k, n))
        for i, cls in enumerate(self.classes_):
            binary_y = (y == cls).astype(float)
            theta = np.zeros(n)
            for _ in range(self.max_iter):
                theta -= self.learning_rate * self.gradient(theta, X, binary_y)
            all_theta[i] = theta
        self.classifiers = all_theta

    def predict(self, X):
        probs = self.sigmoid(X.dot(self.classifiers.T))
        predictions = np.array([self.classes_[np.argmax(probs[i])] for i in range(X.shape[0])])
        return predictions

learning_rates = [0.01, 0.1, 0.5]
lambas = [0, 0.01, 0.1, 1]
results = []
accuracy = []

def macroF1(y_pred, y_true):
    labels = np.unique(y_true)
    f1_scores = []

    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

PCAs = []
componentAmount = [16,32,64,128,256]
for number in componentAmount:
    pcs = PCA_from_Scratch(train_normal, number)
    train_proj = transform(train_normal, pcs)
    test_proj = transform(test_normal, pcs)
    PCAs.append([train_proj, test_proj, number])

for dataset in PCAs:
    for lr in learning_rates:
        for lamb in lambas:
            
            
            modelMC = myMultiClassLogisticRegression(learning_rate=lr, maxiterations=1000, mu=lamb)
            modelMC.fit(dataset[0], train_labels)
            y_pred3 = modelMC.predict(dataset[1])
            accuracy = score(y_pred=y_pred3,y_true=test_labels)
            macro = macroF1(y_pred=y_pred3,y_true=test_labels)            
            print("With PCA Amount:", dataset[2] ,"MultiClass Learning Rate:", lr, "Lambda:", lamb, "Test Accuracy:", accuracy)
        
            modelOA = OneVsAllLogisticRegression(learning_rate=lr, max_iter=1000, _lambda=0.1)
            modelOA.fit(dataset[0], train_labels)
            y_pred2 = modelOA.predict(dataset[1])
            accuracy = score(y_pred=y_pred2,y_true=test_labels)
            macro = macroF1(y_pred=y_pred2,y_true=test_labels)            
            print("With PCA Amount:", dataset[2] ,"One vs All Learning Rate:", lr, "Lambda:", lamb, "Test Accuracy:", accuracy)
