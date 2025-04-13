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
class myNaiveBayes():
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        for w in self.classes:
            X_train_w = X_train[y_train == w]
            P_w = X_train_w.shape[0] / X_train.shape[0] # Prior Densities P(w_j)
            self.priors[w] = P_w
            self.likelihoods[w] = {}
            for x in range(X_train.shape[1]):
                feature_x = X_train_w[:,x]
                mean_x_w = np.mean(feature_x)
                var_x_w = np.var(feature_x)
                self.likelihoods[w][x] = (mean_x_w, var_x_w)

    def gaussianDensity(self,x,mean,var):
        denu = 1.0 / np.sqrt(2.0 * np.pi * (var + 1e-4))
        inexp = -((x - mean) ** 2) / (2 * (var + 1e-4))
        return denu*np.exp(inexp)
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            posteriors = {}
            for c in self.classes:
                log_prior = np.log(self.priors[c])
                log_likelihood = 0
                for i in range(len(x)):
                    mean, var = self.likelihoods[c][i]
                    log_likelihood += np.log(self.gaussianDensity(x[i], mean, var) + 1e-10)
                posteriors[c] = log_prior+log_likelihood
            y_pred.append(max(posteriors, key=posteriors.get))
        return y_pred
mybayes = myNaiveBayes()
mybayes.fit(train_normal,train_labels)
y_pred = mybayes.predict(test_normal)
acc = score(y_pred,test_labels)
mac = macroF1(y_pred,test_labels)
print("Naive Bayes with Just Normalization: accuracy: ", acc, "Macroscore: ", mac)

mybayes = myNaiveBayes()
mybayes.fit(lda_train,train_labels)
y_pred = mybayes.predict(lda_test)
acc = score(y_pred,test_labels)
mac = macroF1(y_pred,test_labels)
print("Naive Bayes with LDA Transformation: accuracy: ", acc, "Macroscore: ", mac)

componentAmount = [16, 32, 64, 128, 256]
principal_comps = []
train_transform = []
test_transform = []
y_pred = []
scorePCA = []
for number in componentAmount:
    pcs = PCA_from_Scratch(train_normal, number)
    principal_comps.append(pcs)
        
    train_proj = transform(train_normal, pcs)
    test_proj = transform(test_normal, pcs)

    mybayes = myNaiveBayes()
    mybayes.fit(train_proj,train_labels)
    y_pred1 = mybayes.predict(test_proj)
    macroscore = macroF1(y_pred1, test_labels)
    normalscore = score(y_pred1, test_labels)

    print( "Naive Bayes with PCA component number: ", number, "and scores are: Macro : ", macroscore,"Accuracy ", normalscore)