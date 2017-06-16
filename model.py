import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
h2o.init()
h2o_df = h2o.load_dataset("prostate.csv")
h2o_df["CAPSULE"] = h2o_df["CAPSULE"].asfactor()
model=H2OGradientBoostingEstimator(distribution="bernoulli",
				   ntrees=100,
				   max_depth=4,
				   learn_rate=0.1)
model.train(y="CAPSULE",
	    x=["AGE","RACE","PSA","GLEASON"],
	    training_frame=h2o_df)
modefile =  model.download_mojo(path = "exp",get_genmodel_jar=True)
print ("Model saved to " + modefile)

