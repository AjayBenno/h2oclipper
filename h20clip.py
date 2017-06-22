import h2o
from clipper_admin import Clipper
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import numpy as np
import sys


print sys.prefix

iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target

print iris_x
print len(iris_x)
dat = np.empty([len(iris_x),5])
# dat = np.array()
for i in range(0,len(iris_x)):
    dat[i] = np.append(iris_x[i],iris_y[i])

h2o.init()
df = h2o.H2OFrame(dat)
df[4] = df[4].asfactor()
model = H2ODeepLearningEstimator()
model.train(x=range(4),y=4,training_frame=df)


def predict_fn(inputs):
    f = h2o.H2OFrame(inputs)
    return model.predict(f)

clipper_client = Clipper("localhost")
clipper_client.start()

clipper_client.register_application("h2o","omodel","doubles","-1.0",100000)
clipper_client.deploy_predict_function("omodel",1,predict_fn,"doubles")
