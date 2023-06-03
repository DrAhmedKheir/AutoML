#A curious library that you may have never heard of is h2o. An in-memory platform for distributed and scalable machine learning, h2o can run on powerful clusters when you need boosted computing power. 
#h2o is a very interesting and diverse library. It contains so many diverse features (ranging from training models to automl capabilities) that it’s easy to get a bit lost when using it, particularly due to the high number of methods and functions one can use with the package. 

#Loading the Data, we used here the trial training dataset (geoSpatialData_linked_fieldData), but we can use also (linked_geoSpatial_AOI) dataset for the predictions.
Dataset2Clean <- read.csv('/Dataset2Clean.csv')
head(Dataset2Clean)
#After loading the data, let’s just do a small check on our data types. If we call the str command 
str(Dataset2Clean)


#Train-Test Split
# h2o has a convenient function to perform train test splits. To get h2orunning, we’ll need to load and initialize our library
library(h2o)
h2o.init()

#When we type h2o.init() , we are setting up a local h2o cluster. By default, h2o will spun all available CPU’s but you can specify a specific number of CPU’s to initialize using nthread
#This is one of the major differences regarding other machine learning libraries in R — to use h2o, we always need to start an h2o cluster. The advantage is that if you have an h2o instance running on a server, you can connect to that machine and use those computing resources without changing the code too much (you only need to point yourinitto another machine)

#Using h2o.splitFrame , we can conveniently create a random training-test split of our data but before that, we need to convert our dataframe into a special object that h2o can recognize

#First learning: h2o can´t deal with normal R dataframes but only with a special type of object h2OFrame, so the step of converting dataframes using as.h2ois mandatory.
#We are now ready to do our train test split with our new data.h2o
Dataset2Clean.h2o <- as.h2o(Dataset2Clean)

Dataset2Clean_split <- h2o.splitFrame(data = Dataset2Clean.h2o, ratios = 0.8, seed = 1234)
training_data <- Dataset2Clean_split[[1]]
test_data <- Dataset2Clean_split[[2]]

#Using h2o.splitFrame , we can immediately divide our dataset into two different h2o frames — ratios define the percentage we want to allocate to our training data and in the function above, we are using 80% of the dataset for training purposes, leaving 20% as an holdout set.
#With our data split between test and train in h2o format, we’re ready to train our first h2o model!

#Training a Model
#Due to its simplicity, the first model we are going to train will be a linear regression. This model can be trained with theh2o.glm function and first, we need to define the target and feature variables:

predictors <- c("NDVIM1", "NDVIM2", "NDVIM3", "NDVIM4", "NDVIM5", "NDVIM6", "MeanNDVI", "EVIM1", "EVIM2", "EVIM3", "EVIM4", "EVIM5", "EVIM6", "MeanEVI", "GCVIM1", "GCVIM2", "GCVIM3", "GCVIM4", "GCVIM5", "GCVIM6", "MeanGCVI", "GNDVIM1", "GNDVIM2", "GNDVIM3", "GNDVIM4", "GNDVIM5", "GNDVIM6", "MeanGNDVI", "WDRVIM1", "WDRVIM2", "WDRVIM3", "WDRVIM4", "WDRVIM5", "WDRVIM6", "MeanWDRVI", "DEM", "Aspect", "Slope", "BD", "Clay", "OM", "Sand", "Silt", "TmaxAvg", "TminAvg", "SRADAvg", "RainAvg")
#predictors <- c("EVIM4", "MeanEVI","DEM", "BD", "Clay", "OM", "TmaxAvg", "TminAvg", "SRADAvg", "RainAvg")

#predictors <- c("MeanNDVI", "EVIM1", "EVIM2", "EVIM3", "EVIM4", "EVIM5", "EVIM6", "MeanEVI", "MeanGCVI", "DEM", "Aspect", "Slope", "BD", "Clay", "OM", "Sand", "Silt", "TmaxAvg", "TminAvg", "SRADAvg", "RainAvg")

response <- "Yield"
response


Dataset2Clean_model <- h2o.glm(x = predictors,
                             y = response,
                             training_frame = training_data)

#We have our model ready! Let’s compare our predictions against the real value on the test set— we can convinently use the h2o.predict function to get the predictions from our model:

test_predict <- h2o.predict(object = Dataset2Clean_model, 
                            newdata = test_data)
#And then, we can cbind our predictions with the cnt from the test set:
predictions_x_real <- cbind(
  as.data.frame(test_data$Yield),
  as.data.frame(test_predict)
)
#let’s apply some regularization inside the h2o.train using the alpha parameter:

Dataset2Clean_model_regularized <- h2o.glm(x = predictors,
                                         y = response,
                                         training_frame = training_data,
                                         alpha = 1)
#In h2o.glm ,alpha=1 represents Lasso Regression. It doesn’t seem that our model improved that much, and we probably need to do some more feature engineering or try other arguments with the linear regression

#Evaluating our Models
Dataset2Clean_model <- h2o.glm(x = predictors,
                             y = response,
                             training_frame = training_data,
                             validation_frame = test_data)

h2o.rmse(Dataset2Clean_model, train=TRUE, valid=TRUE)

h2o.r2(Dataset2Clean_model, train=TRUE, valid=TRUE)
#In this example, I am following a regression problem but, of course, you also have classification models and metrics available. For all metrics available in h2o, please uncomment the next cell:
# retrieve the mse value:

#mse_basic <- h2o.mse(geoSpatialData2_gbm)
#mse_basic

# retrieve the mse value for both the training and validation data:
#mse_basic_valid <- h2o.mse(geoSpatialData2_gbm, train = TRUE, valid = TRUE, xval = FALSE)
#mse_basic_valid
## retrieve the mae value:
#mae_basic <- h2o.mae(geoSpatialData2_gbm)
#mae_basic
## retrieve the mae value for both the training and validation data:
#mae_basic_valid <- h2o.mae(geoSpatialData2_gbm, train = TRUE, valid = TRUE, xval = FALSE)
#mae_basic_valid
#Classification
#H2O-3 calculates regression metrics for classification problems. The following additional evaluation metrics are available for classification models:
#Gini Coefficient
#Absolute MCC (Matthews Correlation Coefficient)
#F1
#F0.5
#F2
#Accuracy
#Logloss
#AUC (Area Under the ROC Curve)
#AUCPR (Area Under the Precision-Recall Curve)
#Kolmogorov-Smirnov (KS) Metric
 
  
#It’s a bit expected that our linear regression isn’t performing that well — we haven’t performed any feature engineering and we are probably violating too many linear regression assumptions.
#But, if we can train simple linear regressions, we can probably train other types of models in h2o ,
#right? That’s right ! Let’s see that in the next section

#More Model Examples
#If we change the h2o function associated with the training process, we will fit other types of models. 
#Let’s train a random forest by calling h2o.randomForest :
Dataset2Clean_rf <- h2o.randomForest(x = predictors,
                                   y = response,
                                   ntrees = 25,
                                   max_depth = 5,
                                   training_frame = training_data,
                                   validation_frame = test_data)

#I’m setting two hyperparameters for my Random Forest on the function call:
  
#1-ntrees that sets the number of trees in the forest.
#2-maxdepth that sets the maximum deepness of each tree.
#If you need, you can find all the tweakable parameters by calling > help(h2o.randomForest) on the R console
#Metric function
h2o.rmse(Dataset2Clean_rf, train=TRUE, valid=TRUE)
h2o.r2(Dataset2Clean_rf, train=TRUE, valid=TRUE)

#Notice that our code practically didn’t change. 
#The only thing that was modified was the model we fed into the first argument. 
#This makes these metric functions highly adaptable to new models, as long as they are trained inside the h2o framework
#H2O supports the following supervised algorithms:
#1- AutoML: Automatic Machine Learning
#2- Cox Proportional Hazards (CoxPH)
#3- Deep Learning (Neural Networks)
#4- Distributed Random Forest (DRF)
#5- Generalized Linear Model (GLM)
#6- Isotonic Regression
#7- ModelSelection
#8- Generalized Additive Models (GAM)
#9- ANOVA GLM
#10-Gradient Boosting Machine (GBM)
#11-Naïve Bayes Classifier
#12-RuleFit
#13-Stacked Ensembles
#14-Support Vector Machine (SVM)
#15-Distributed Uplift Random Forest (Uplift DRF)
#16-XGBoost
###let’s fit a Neural Network, using h2o.deeplearning:
nn_model <- h2o.deeplearning(x = predictors,
                             y = response,
                             hidden = c(7,7,4,5),
                             epochs = 2000,
                             train_samples_per_iteration = -1,
                             reproducible = TRUE,
                             activation = "Rectifier",
                             seed = 23123,
                             training_frame = training_data,
                             validation_frame = test_data)
h2o.r2(nn_model, train=TRUE, valid=TRUE)

#NN with GridSearch hyperparameter tuning
nn_params <- list(activation = "RectifierWithDropout",
                  epochs = 10000,
                  epsilon = c(1e-6, 1e-7, 1e-8, 1e-9),
                  hidden = c(7,7,4,5),
                  hidden_dropout_ratios = c(0.1,0.2,0.3,0.4,0.5),
                  rho = c(0.9,0.95,0.99))

nn_grid <- h2o.grid("deeplearning", 
                    x = predictors, 
                    y = response,
                    grid_id = "nn_grid",
                    training_frame = training_data,
                    validation_frame = test_data,
                    seed = 1,
                    hyper_params = nn_params)

h2o.getGrid(grid_id = "nn_grid",
            sort_by = "r2",
            decreasing = TRUE)
#hidden is a very important argument in the h2o.deeplearning function. It takes a vector that will represent the number of hidden layers and neurons we will use on our neural network. In our case, we are using c(7,7,4,5) , 4 hidden layers with 6, 6, 4 and 7 nodes each.
#Another cool feature of h2o is that we can do hyperparameter tuning really smoothly — let’s see how, next.

#HyperParameter Tuning
#Performing hyperparameter search is also super simple in h2o,
#In the grid example, we’ll do a search on both parameters plus min_rows . We can do that by using the h2o.grid function:
# Grid Search 
rf_params <- list(ntrees = c(2, 5, 10, 15),
                  max_depth = c(3, 5, 9),
                  min_rows = c(5, 10, 100))
# Train and validate a grid of randomForests
rf_grid <- h2o.grid("randomForest", 
                    x = predictors, 
                    y = response,
                    grid_id = "rf_grid",
                    training_frame = training_data,
                    validation_frame = test_data,
                    seed = 1,
                    hyper_params = rf_params)
#We start by declaring rf_params that contain the list of values we will use on our grid search and then, we pass that grid into hyper_params argument in the h2o.grid . What h2owill do is train and evaluate every single combination of hyperparameters available.
#The h2o.getGrid function gives us a summary of the best hyperparameters according to a specific metric. In this case, we chose r2 , but other metrics such as the RMSE or MSE also work. Let’s look at the top 5 results of our grid search:

h2o.getGrid(grid_id = "rf_grid",
            sort_by = "r2",
            decreasing = TRUE)
#Our best model was the one that had a max_depth of 9, a min_rows of 5 and 15 ntrees — this model achieved an r2 of 0.7669.

#The cool part? You can expand this grid to any hyperparameter available in ?h2o.randomForest or to any model available in the documentation of h2o , opening up an endless amount of possibilities with the same function.

#######################################################

#AutoML Features########################################
#If you need a quick and raw way to look at the way different models perform on your dataset, h2o also has an interesting automl routine:
aml <- h2o.automl(x = predictors, 
                  y = response,
                  training_frame = training_data,
                  validation_frame = test_data,
                  max_models = 20,
                  seed = 1)

#The max_models argument specify the maximum number of models to be tested on a specific automl ensemble. Keep in mind that the automl routine can take a while to run, depending on your resources.

#We can access the top models of our routine by checking the aml@leaderboard
aml@leaderboard
#By the table above, we can see that a Stacked Ensemble model was the winner (at least, in terms of rmse). We can also get more information on the best model by calling:
h2o.get_best_model(aml)
#The result of h2o.get_best_model(aml) returns more information about the model that achieved the best score on our automlroutine. 
#By the snippet above, we know that our ensemble aggregates the result of:

#Depending on your use case, the automl routine can be a quick, dirty way, to understand how ensembles and single models behave on your data, giving you hints on where to go next. 
#For example, helping you on the decision if more complex models are the way to go or if you will probably need to acquire more training data / features.


#Explainability
#Finally, let’s take a look into some of h2o’s explainability modules. In this example, we’ll use the random forest model we’ve trained above:
Dataset2Clean_rf <- h2o.randomForest(x = predictors,
                                   y = response,
                                   ntrees = 25,
                                   max_depth = 5,
                                   training_frame = training_data,
                                   validation_frame = test_data)
#Just like most machine learning libraries, we can grab the variable importance plot directly with the h2o interface:
par(mar = c(1, 1, 1, 1))#Expand the plot layout pane
h2o.varimp_plot(Dataset2Clean_rf)

#By the importance plot, we notice that Ptot and P-0 are the most important variables for our trained random forest. Calling varimp_plot immediately shows the importance plot for a specific model, without the need to configure anything else.

#Single variable importances are not the only available explainability models in h2o — we can also check shap values quickly:
h2o.shap_summary_plot(Dataset2Clean_rf, test_data)
#Voilà! By the shap_summary_plot, we understand the direction of the relationship between our features and target. For instance:

#Lower DEM explain low yield.
###################################Lets use different dataset, one for training and another for predicting
#Lets use different datasets, one for traning (tarin.csv) and other for testing (test.csv)

h2o.init()

train <-h2o.importFile("D:/ICARDA/ICARDA publications/H2OAutoML/TrainTestDatset/train.csv")
test <-h2o.importFile("D:/ICARDA/ICARDA publications/H2OAutoML/TrainTestDatset/test.csv")

#identify predictors and respons
y <-"Yield"



x <- setdiff(names(train), y)

aml <-h2o.automl(x=x,y=y,
                 training_frame=train,
                 max_models = 20,
                 seed = 1)

#View AutoML leaderboard
lb <-aml@leaderboard

print(lb,n=nrow(lb))

#The leader model stored here
aml@leader

#To generate predictions on a test set
#directly on the H2oAutoML object or on the leader model
#object directly

pred <-h2o.predict(aml,test)
pred

class(pred)#to know the data type to enable saving it 

pred_hf <- as.h2o(pred)
h2o.exportFile(pred_hf, path = "D:/ICARDA/ICARDA publications/H2OAutoML/TrainTestDatset/pred.csv")




