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
lda = LDA()
lda.fit(train_normal,train_labels)
lda_train = lda.transform(train_normal)
lda_test = lda.transform(test_normal)
class myKNearesNeighbour():
    def __init__(self, k):
        self.k = k
    def fit(self, X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        self.X_test = X_test
        y_pred = np.zeros(self.X_test.shape[0])
        for i, image in enumerate(X_test):
            neighbours = self.findNeighbours(image)
            y_pred[i] = self.mode(neighbours)
        return y_pred
    def findNeighbours(self, image):
        sortedList = np.argsort(self.Euclidean(image)) 
        y_train_sorted = self.y_train[sortedList]
        return y_train_sorted[:self.k]
    def mode(self, vector):
        values, counts = np.unique(vector, return_counts=True)
        return values[np.argmax(counts)]
    def Euclidean(self,image):
        distances = np.sqrt(np.sum((self.X_train - image) ** 2, axis=1))
        return distances
n_neighbours = [3,5,7,9]
scores = []
macrolist = []
for k in n_neighbours:
    KNN = myKNearesNeighbour(k)
    KNN.fit(lda_train,train_labels)
    y_pred = KNN.predict(lda_test)
    result = score(y_pred, test_labels)
    scores.append(result)
    macrolist.append(macroF1(y_pred,test_labels))
    print(k , " N neighbours with LDA ", result)

for k in n_neighbours:
    KNN = myKNearesNeighbour(k)
    KNN.fit(train_normal,train_labels)
    y_pred = KNN.predict(test_normal)
    result = score(y_pred, test_labels)
    scores.append(result)
    macrolist.append(macroF1(y_pred,test_labels))
    print(k , " N neighbours only with normalized " , result)

componentAmount = [16, 32, 64, 128, 256]
principal_comps = []
train_transform = []
test_transform = []
y_pred = []
scorePCA = []
for k in n_neighbours:
    for number in componentAmount:
        pcs = PCA_from_Scratch(train_flat, number)
        principal_comps.append(pcs)
        
        train_proj = transform(train_flat, pcs)
        test_proj = transform(test_flat, pcs)
        train_transform.append(train_proj)
        test_transform.append(test_proj)
        
        KNN_PCA = myKNearesNeighbour(k)
        KNN_PCA.fit(train_proj, train_labels)
        pred = KNN_PCA.predict(test_proj)
        y_pred.append(pred)
        macroscore = macroF1(y_pred,test_labels)
        normalscore = score(pred, test_labels)

        print(k, " with PCA component number: ", number, "and scores are: Macro : ", macroscore,"Accuracy ", normalscore)

        
