import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        numSamples, numFeatures = X.shape
        self.classes = np.unique(y)
        numClasses = len(self.classes)

        self.mean = np.zeros((numClasses, numFeatures), dtype=np.float64)
        self.var = np.zeros((numClasses, numFeatures), dtype=np.float64)
        self.prior = np.zeros(numClasses, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / numSamples

    def predict(self, X):
        yPred = [self._predict(x) for x in X]
        return np.array(yPred)
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            classCond = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + classCond
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, classIdx, x):
        mean = self.mean[classIdx]
        var = self.var[classIdx]
        numerator = np.exp(-((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


