from sklearn.svm import LinearSVC

import numpy as np

class TresholdLinearClassifier():

    def __init__(self, ClassClass, treshold='treshold0', **keywords):
        '''
        :param treshold:
        :param ClassClass: python class implementring a linear classifier
        :param keywords:
        :return:
        '''
        self.classifier = ClassClass(**keywords)
        self.treshold = treshold
        self.targetLabel = 1

    def get_params(self, deep=True):
        return { 'treshold' : self.treshold , 'classifier' : self.classifier }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        print 'fit'
        self.classifier.fit(X, y)
        print 'intercept: %.3f' % self.classifier.intercept_
        avg = 0.0
        for i in range(len(y)):
            if y[i] == self.targetLabel:
                avg += self.classifier.decision_function(X[i])
        avg /= len(y)
        #if avg < 0

    def predict(self, X):
        result = self.classifier.predict(X)
        y = np.zeros(len(X))
        return result

    def testPredict(self, X, true):
        print 'testPredict'
        result = self.classifier.predict(X)
        for i in range(len(X)):
            if true[i] == self.minClass:
                print 'true %d , pred %d , decision %.3f' % \
                      (true[i], result[i], self.classifier.decision_function(X[i]))