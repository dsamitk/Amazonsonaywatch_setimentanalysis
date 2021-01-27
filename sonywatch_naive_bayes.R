#NAive Bayes Algorithm
#-----------------------Amazon dataset------------------------------------#
#Read the data
setwd("E:\\Text mining\\sentiment analysis")
analysis<-read.csv("review_sentiment.csv",stringsAsFactors=FALSE)

#Split the data into training and test
set.seed(2000)
sampling<-sort(sample(nrow(analysis), nrow(analysis)*.7))

length(sampling)

head(analysis)
names(analysis)
train_tweets = analysis[sampling,]
test_tweets = analysis[-sampling,]

prop.table(table(train_tweets$sentiment))
prop.table(table(test_tweets$sentiment))

#-----------------------Naive Bayes algorithm ----------------------------#

#Create induvidual matrices for training and testing datasets
mtrain<-as.matrix(train_tweets)
mtest<-as.matrix(test_tweets)

#Building Document term matrices for training and testing data

library(RTextTools)
library(e1071)
train_matrix= create_matrix(mtrain[,2], language="english",removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0, 
                            removeStopwords=TRUE, stripWhitespace=TRUE, toLower=TRUE) 
test_matrix= create_matrix(mtest[,2], language="english",removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0, 
                           removeStopwords=TRUE, stripWhitespace=TRUE, toLower=TRUE) 

#Input to naive bayes algorithm has to be a matrix with categorical values,
#not numeric

#Convert DTM to a 1/0 matrix
conversion<-function(A)
{
  A<-ifelse(A>0,1,0)
  A<-factor(A,levels=c(0,1),labels=c("No","Yes"))
  return(A)
}

library(tm)
View(inspect(train_matrix[1:10,1:10]))

mat_train<-apply(train_matrix,MARGIN=2,conversion)

View(mat_test[1:10,1:10])

mat_test<-apply(test_matrix,MARGIN=2,conversion)

#Train the model
library(e1071)

#Input the training matrix and the Dependent Variable

classifier = naiveBayes(mat_train,as.factor(mtrain[,3]))

# Validation
predicted = predict(classifier,mat_test) #predicted
length(predicted)
#Model Performance Metrics
install.packages("gmodels")
library(gmodels)
install.packages("caret")
library(caret)

#Confusion Matrix
CrossTable(predicted,mtest[,3],prop.chisq=FALSE,prop.t=FALSE,dnn=c('predicted','actual'))

#accuracy of the model 
recall_accuracy(as.numeric(as.factor(test_tweets[,3])), as.numeric(predicted))

#Using Caret Package
confusionMatrix(predicted,mtest[,3])
#get values such as KAppa,Accuracy,Sensitivity/Recall,Specificity,PPV,NPV



