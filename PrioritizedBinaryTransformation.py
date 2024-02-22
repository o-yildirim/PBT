import numpy as np
from river import optim
from river import linear_model
from river.tree import HoeffdingTreeRegressor
from sklearn.decomposition import PCA

class PrioritizedBinaryTransformation:
        def __init__(self, n_labels,t):
            self.n_labels = n_labels
            self.model = self.get_model()
            self.t = t #Column count of the matrix, (L x t).
            self.n = 0 #Number of processed data.
            self.M = [] #L X t matrix, initially empty.
            self.discriminative_labels = list(range(0, self.n_labels))



        def learn_one(self, X, y):
            y_temp = list(y.values())
            y_o = []
            for index in self.discriminative_labels:
                y_o.append(y_temp[index])

            y_ = self.transform(y_o)
            self.model.learn_one(X, y_)

            self.M = np.append(self.M,[np.array(list(y.values())).astype(int)])
            self.n+=1

            #Add y to the matrix M as a column.
            if self.n % self.t == 0:
                #Compute PCA of the matrix M and get the most discriminative labels.
                discriminative_labels = self.compute_PCA()
                discriminative_labels = np.flip(discriminative_labels, axis=0)
                self.discriminative_labels = discriminative_labels


                #Reset the matrix.
                self.M = []


        def compute_PCA(self):
            # Reshape matrix for PCA.
            self.M = self.M.reshape((self.t, self.n_labels))

            #Perform PCA.
            pca = PCA(n_components=min(self.n_labels,self.t))
            pca.fit(self.M)

            #Find most discriminative indices.
            most_discriminative_indices = np.argsort(np.abs(pca.components_)).flatten()[-self.n_labels:]
            return most_discriminative_indices

        def predict_one(self, X):
            y_pred_temp = np.round(self.model.predict_one(X)).astype(np.int32)


            y_pred_temp = self.binarize(y_pred_temp)
            y_pred = [0] * self.n_labels
            for i in range(self.n_labels):
                y_pred[self.discriminative_labels[i]] = y_pred_temp[i]

            return np.array(y_pred).astype(np.int32)


        def get_model(self):
            return HoeffdingTreeRegressor(
                leaf_model=linear_model.LinearRegression(optimizer=optim.Adam(lr=1e-5), intercept_lr=0.5,
                                                         loss=optim.losses.Squared()))

        def transform(self, Y):
            Y_ = 0
            for bit in Y:
                Y_ = (Y_ << 1) | bit
            return Y_

        def binarize(self, Y):
            if Y > 2 ** self.n_labels - 1:
                y_ = [1] * self.n_labels
            y_ = np.array(list(map(int, list(np.binary_repr(Y, self.n_labels))))).astype(np.int32)
            if len(y_) > self.n_labels:
                y_ = np.ones(self.n_labels).astype(np.int32)
            return np.array(y_).astype(np.int32)