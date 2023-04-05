
#------------------------------------------------------------------------------#
#                   MGMT-715-676 BUSINESS CONSULTING                           #
#                                Group 4                                       #
#           Amaan Ghauri, Shivam Pande, Utsav Pradhan, Amrut Prasade           #
#------------------------------------------------------------------------------#


#------------------------------------------#
#####       Data Preparation           #####
#------------------------------------------#

## Clearing Environment ##

rm(list=ls())

## Loading Libraries ##

library("caret")
library("rpart")
library("rpart.plot")
library("NeuralNetTools")
library("e1071") 
library("tidyverse")
library("ggplot2")
library("viridis")


## Setting Working Directory

setwd("D:/Course/Drexel/Term 4 - Summer 2022/MGMT 715 - Business Consulting/Final Report/R")


## Reading target file, importing blank spaces as NA

mtg <- read.csv(file = "loan_mortgage.csv",
                na.strings = c("", " "),
                stringsAsFactors = FALSE)


## Creating an aggregated Sample:

### Aggregating by Max Time
mtga = aggregate(x = mtg, 
                 by = list(mtg$MID), 
                 FUN = max)


mtga <- mtga[,-1] ## omitting the "Group 1" column


### Creating new variables

mtga$avg_LTV <- aggregate(mtg$LTV, 
                          list(mtg$MID), mean) [,2]

mtga$sd_LTV <- aggregate(mtg$LTV, 
                         list(mtg$MID), sd)[,2]

mtga$avg_balance <- aggregate(mtg$balance, 
                              list(mtg$MID), mean)[,2]

mtga$sd_balance <- aggregate(mtg$balance, 
                             list(mtg$MID), sd)[,2]

mtga$avg_interest_rate <- aggregate(mtg$interest_rate, 
                                    list(mtg$MID),  mean)[,2]

mtga$sd_interest_rate <- aggregate(mtg$interest_rate, 
                                   list(mtg$MID), sd)[,2]

mtga$avg_hpi <- aggregate(mtg$hpi, 
                          list(mtg$MID), mean)[,2]

mtga$sd_hpi <- aggregate(mtg$hpi, 
                         list(mtg$MID), sd)[,2]

mtga$avg_gdp <- aggregate(mtg$gdp, 
                          list(mtg$MID), mean)[,2]

mtga$sd_gdp <- aggregate(mtg$gdp, 
                         list(mtg$MID), sd)[,2]

mtga$avg_uer <- aggregate(mtg$uer, 
                          list(mtg$MID), mean)[,2]

mtga$sd_uer <- aggregate(mtg$uer, 
                         list(mtg$MID), sd)[,2]


## Aggregating further by:

### first observation time is equal to the mortgage origination time

mtga <- mtga[mtga$first_ob_time == mtga$origin_time, ]

### Dropping redundant variables

mtga <- mtga[ , -c(4,6:11,19,20)]

### Conversion to Alternate Variable Types


### Setting up target variable
mtga$status_time <- factor(mtga$status_time)


### Nominal categorical variables as facs

facs <-  c("RE_Type","investor")


## no ordinal variables

### Numerical Variable as nums

nums <- c("maturity_time","balance_orig_time", "LTV_orig_time", "Interest_Rate_orig_time", 
           "hpi_orig_time", "avg_LTV", "sd_LTV", "avg_balance", "sd_balance", 
           "avg_interest_rate", "sd_interest_rate", "avg_hpi", "sd_hpi", 
           "avg_gdp", "sd_gdp", "avg_uer", "sd_uer")

### Checking the unique values that the variables can take on.

### For all facs variables
lapply(X = mtga[ ,facs],
       FUN = unique)


### Converting facs variables to factors
mtga[ ,facs] <- lapply(X = mtga[ , facs], 
                       FUN = factor)


vars<-c(facs,nums)

#------------------------------------------#
#####       Data Exploration           #####
#------------------------------------------#

## Rechecking the structure
str(mtga)

## Visualizing our Target Variable

par(mfrow = c(1,1))
mttab <- table(mtga$status_time)
mtbpt <- barplot(mttab,
                 main = "Bar Plot: Status Time",
                 xlab="Status Time",
                 ylab = "Count",
                 col = c("gray19",'coral3','darkslategray3'),
                 names.arg = c("Neither", "Default", "Payoff"))

text(mtbpt, mttab/2, labels = round(mttab, digits = 2), col = "white", )




#### We can see that there is Class Imbalance
#### in our target variable. This imbalance
#### will be corrected later in the models.

## Statistical Summary of Numerical variables from mtga dataframe
sapply(X = mtga[,nums], 
       FUN = summary)

## Standard deviation of numerical variables
sapply(X = mtga[ ,nums], 
       FUN = sd)


## Exploring all categorical variables through 1-way frequency table
lapply(X = mtga[ ,c(facs)], FUN = table)


## Exploring all categorical variables through 1-way frequency table
lapply(X = mtga[ ,c(facs)], FUN = table)




#------------------------------------------#

## OUTLIERS

par(mfrow = c(1,1))

outs <- sapply(mtga[,c(1:4,7:11,13:23)], function(x) which(abs(scale(x)) > 3))
outs


### Average Balance
boxplot(x = mtga[,c(15)], 
        main="Box Plot: Avg Balance Outliers",
        horizontal = TRUE)

### SD Balance
boxplot(x = mtga[,c(16)], 
        main="Box Plot: SD Balance Outliers",
        horizontal = TRUE)

### Average HPI 
boxplot(x = mtga[,c(19)], 
        main="Box Plot: Avg HPI Outliers",
        horizontal = TRUE)


#------------------------------------------#
#####       Data Pre-processing        #####
#------------------------------------------#

## Checking for missing variables

mtga[!complete.cases(mtga),]

na_rows <- rownames(mtga)[!complete.cases(mtga)]
na_rows
summary(mtga)


#### We can see that there are NA values in Standard Deviation. 
#### Due to events with only 1 observation among the IDs
#### there are no SD values. Here, the NA values are rather
#### 0 values than missing values.

### Impute in NAs

#### Making a copy of the dataframe
mg <- mtga

# Impute missing values with 0
mg[is.na(mg)] <- 0

mg[!complete.cases(mg),]

summary(mg)

#write.csv(mg,"loan_mg.csv") # creating a new dataset for convenience

#------------------------------------------#
#####         Decision Trees           #####
#------------------------------------------#

## Training and Testing

### Splitting the data into training and 
### testing sets using an 80/20 split rule

## Initialize random seed
set.seed(2527) 

## Create list of training indices
sub <- createDataPartition(y = mg$status_time, # target variable
                           p = 0.8,  # % of split in training
                           list = FALSE)

## Subset the transformed data 
## to create the training (train) and testing (test) datasets

train <- mg[sub, ] # create train dataframe
test <- mg[-sub, ] # create test dataframe

## Oversampling to correct for Class Imbalance

train <- upSample(x = train, # predictors
                      y = train$status_time, # target
                      yname = "Status Time") # name of y variable



mg.rpart <- rpart(formula = status_time ~ ., # Y ~ all other variables in dataframe
                  data = train[ ,c(vars, "status_time")], # include only relevant variables
                  method = "class")

### Basic output of our Decision Tree model
mg.rpart

### Checking for Variable Importance

mg.rpart$variable.importance

## Tree Plots

### Using the prp() fn to plot rpart object (mg.rpart)

prp(x = mg.rpart, # rpart object
    extra = 2) # include proportion of correct predictions

## Training Performance

### Using predict() function to generate 
### class predictions for training set

base.trpreds <- predict(object = mg.rpart, # DT model
                        newdata = train, # training data
                        type = "class") # class predictions

### Using confusionMatrix() function from the caret package 
### to obtain a confusion matrix and obtain performance measures
### for our model applied to the testing dataset (train).

DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$status_time, # actual
                                 positive = "1",
                                 mode = "everything")
DT_train_conf


## Testing Performance
### Using predict() function to generate 
### class predictions for testing set

base.tepreds <- predict(object = mg.rpart, # DT model
                        newdata = test, # testing data
                        type = "class")


### Using confusionMatrix() function from the caret package 
### to obtain a confusion matrix and obtain performance measures
### for our model applied to the testing dataset (test).


DT_test_conf <- confusionMatrix(data = base.tepreds, # predictions
                                reference = test$status_time, # actual
                                positive = "1",
                                mode = "everything")
DT_test_conf



### Comparing the performance on the training and testing. 
### We can use the cbind() function to compare side-by-side.

# Overall
cbind(Training = DT_train_conf$overall,
      Testing = DT_test_conf$overall)

# Class-Level
cbind(Training = DT_train_conf$byClass,
      Testing = DT_test_conf$byClass)

#------------------------------------------#

### 2. Hyperparameter Tuning Model

# Using the train() function in the caret 
# package

# We will perform a grid search for the 
# optimal cp value.

# We want to tune the cost complexity 
# parameter, or cp. We choose the cp 
# that is associated with the smallest 
# cross-validated error (highest accuracy)

# We will search over a grid of values
# from 0 to 0.05. We use the expand.grid()
# function to define the search space

grids <- expand.grid(cp = seq(from = 0,
                              to = 0.05,
                              by = 0.005))
grids

# First, we set up a trainControl object
# (named ctrl) using the trainControl() 
# function in the caret package. We specify 
# that we want to perform 10-fold cross 
# validation, repeated 3 times and specify
# search = "grid" for a grid search. We use 
# this object as input to the trControl 
# argument in the train() function below.

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     search = "grid")


# Next, we initialize a random seed for 
# our cross validation

set.seed(2527)

# Then, we use the train() function to
# train the DT model using 5-Fold Cross 
# Validation (repeated 3 times). We set
# tuneGrid equal to our grid search
# objects, grids.

DTFit <- train(form = status_time ~ ., # use all variables in data to predict delay
               data = train[ ,c(vars, "status_time")], # include only relevant variables
               method = "rpart", # use the rpart package
               trControl = ctrl, # control object
               tuneGrid = grids) # custom grid object for search

# We can view the results of our
# cross validation across cp values
# for Accuracy and Kappa. The output
# will also identify the optimal cp.

DTFit # all results

DTFit$results[DTFit$results$cp %in% DTFit$bestTune,] #best result


# We can plot the cp value vs. Accuracy

plot(DTFit)

# We can view the confusion matrix showing
# the average performance of the model
# across resamples

confusionMatrix(DTFit)

# Decision Trees give us information 
# about Variable Importance. We can use
# the best fit object from the caret
# package to obtain variable importance
# information 

DTFit$finalModel$variable.importance


## Tuned Model Performance

## Training Performance

### We use the predict() function to generate 
### class predictions for our training data set

tune.trpreds <- predict(object = DTFit,
                        newdata = train,)


DT3_trtune_conf <- confusionMatrix(data = tune.trpreds, # predictions
                                   reference = train$status_time, # actual
                                   positive = "1",
                                   mode = "everything")
DT3_trtune_conf


## Testing Performance

### We use the predict() function to generate class 
### predictions for our testing data set

tune.tepreds <- predict(object = DTFit,
                        newdata = test)


DT3_tetune_conf <- confusionMatrix(data = tune.tepreds, # predictions
                                   reference = test$status_time, # actual
                                   positive = "1",
                                   mode = "everything")
DT3_tetune_conf


## Goodness of Fit

## Checking for Overall Performance

cbind(Training = DT3_trtune_conf$overall,
      Testing = DT3_tetune_conf$overall)

## Checking for Class-Level Performance

cbind(Training = DT3_trtune_conf$byClass,
      Testing = DT3_tetune_conf$byClass)

## We can also create an excel file to easily access
## the results of Model performance.
# 
# dt3_over<-cbind(Training = DT3_trtune_conf$overall,
#                 Testing = DT3_tetune_conf$overall)
# 
# 
# dt3_classlvl<-cbind(Training = DT3_trtune_conf$byClass,
#                     Testing = DT3_tetune_conf$byClass)
# 
# write.csv(dt3_over,"dt3_over.csv")
# write.csv(dt3_classlvl,"dt3_classlevel.csv")

#------------------------------------------#


#------------------------------------------#
#####           Naive Bayes            #####
#------------------------------------------#

### Standardization for Naive Bayes
### Yeo-Johnson and Center Scaling

cen_yjcs <- preProcess(x = mg[, vars],
                       method = c("YeoJohnson", "center", "scale"))



mg_yj <- predict(object = cen_yjcs,
                    newdata = mg)

#------------------------------------------#

# Training and Testing

## Splitting the data into training and 
## testing sets using an 80/20 split rule

## Initialize random seed
set.seed(2527)

## Create list of training indices
sub_nb <- createDataPartition(y = mg_yj$status_time, # target variable
                              p = 0.80, # % in training
                              list = FALSE)

## Subset the transformed data
## to create the training (train)
## and testing (test) datasets
train_nb <- mg_yj[sub_nb, ] # create train dataframe
test_nb <- mg_yj[-sub_nb, ] # create test dataframe


## oversampling our Target Variable for Class Imbalance

train_nbs <- upSample(x = train_nb, # predictors
                      y = train_nb$status_time, # target
                      yname = "Status Time") # name of y variable


## Visualizing the over sampled Target Variable

par(mfrow = c(1,2))

nbtab <- table(train_nb$status_time)
bpt <- barplot(nbtab,
               main = "Bar Plot: Status Time Class Imbalance",
               xlab="Status Time",
               ylab = "Count",
               col = c("gray19",'coral3','darkslategray4'),
               names.arg = c("Neither", "Default", "Payoff"))

text(bpt, nbtab/2, labels = round(nbtab, digits = 2), col = "white", )

nbtab2 <- table(train_nbs$status_time)
bpt2 <- barplot(nbtab2,
                main = "Bar Plot: Status Time Oversampled",
                xlab="Status Time",
                ylab = "Count",
                col = c("gray19",'coral3','darkslategray4'),
                names.arg = c("Neither", "Default", "Payoff"))

text(bpt2, nbtab2/2, labels = round(nbtab2, digits = 2), col = "white", )


#------------------------------------------#

# Analysis


## Looking for zero probability categories
## To determine if we need to use Laplace smoothing

aggregate(train_nbs[ ,c(facs)],
          by = list(train_nbs$status_time),
          FUN = table)

## Applying Laplace Smoothing

nb_mod <- naiveBayes(x = train_nbs[ ,vars],
                     y = train_nbs$status_time,
                     laplace = 1) 
nb_mod

#------------------------------------------#

# Model Performance & Fit

## Training Performance

## Comparing the training and testing performance
## to assess the goodness of fit of the model

## Using the predict() function to obtain the class predictions
## for the training set
nb.train <- predict(object = nb_mod, # NB model
                    newdata = train_nbs[ ,vars], # predictors
                    type = "class")


head(nb.train)

## Using the confusionMatrix() function
## confusion matrix and obtain performance

tr_nb_conf <- confusionMatrix(data = nb.train, # predictions
                              reference = train_nbs$status_time, # actual
                              positive = "1",
                              mode = "everything")
tr_nb_conf


## Testing Performance

## Using the predict() function to obtain the class predictions
## for the training set

nb.test <- predict(object = nb_mod, # NB model
                   newdata = test_nb[ ,vars], # predictors
                   type = "class")

## Using the confusionMatrix() function
## confusion matrix and obtain performance

te_nb_conf <- confusionMatrix(data = nb.test, # test predictions
                              reference = test_nb$status_time, # actual
                              positive = "1",
                              mode = "everything")
te_nb_conf


## Checking for Overall Performance

te_nb_conf$overall[c("Accuracy", "Kappa")]

## Checking for Class-Level Performance

te_nb_conf$byClass


## Goodness of Fit


## Overall
cbind(Training = tr_nb_conf$overall,
      Testing = te_nb_conf$overall)

## Class-Level
cbind(Training = tr_nb_conf$byClass,
      Testing = te_nb_conf$byClass)


## We can also create an excel file to easily access
## the results of Model performance.
# 
# nb_over<-cbind(Training = tr_nb_conf$overal,
#                 Testing = te_nb_conf$overall)
# 
# 
# nb_classlvl<-cbind(Training = tr_nb_conf$byClass,
#                     Testing = te_nb_conf$byClass)
# 
# write.csv(nb_over,"nb_over.csv")
# write.csv(nb_classlvl,"nb_classlevel.csv")

#save.image(file = "Mortgage_Group4.RData")

#####--------------End--------------#####

