import numpy as np

class NaiveBayes:

    # Taking all the features into account and training the model
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
            self.var[idx, :] = X_c.var(axis=0) + 1e-9 # Adding smoothing here as well to keep accuracy
            self.prior[idx] = X_c.shape[0] / numSamples

    # Predicting the class of the input data
    def predict(self, X):
        yPred = [self._predict(x) for x in X]
        return np.array(yPred)
    
    # Helper function to predict the class of the input data
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            pdf_values = self._pdf(idx, x)
            pdf_values = np.where(pdf_values == 0, 1e-9, pdf_values)  # Smoothing, otherwise division by 0
            classCond = np.sum(np.log(pdf_values))
            posterior = prior + classCond
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    # Helper function to calculate the probability density function
    def _pdf(self, classIdx, x):
        mean = self.mean[classIdx]
        var = self.var[classIdx]
        numerator = np.exp(-((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        pdf_values = numerator / denominator


        return pdf_values


