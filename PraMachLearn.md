# Prediction Assignment Writeup

For this assignment we will analyze the provided data to determine what activity an individual performed. To do this we will make use of caret and randomForest, this will allow us to generate correct answers for each of the 20 test data cases provided after this assignment.



```r
suppressWarnings(library(Hmisc))
suppressWarnings(library(caret))
suppressWarnings(library(randomForest))
suppressWarnings(library(foreach))
suppressWarnings(library(doParallel))
```

We will first load the data both from the provided training and test data provided 

Some values contained a "#DIV/0!" that need to be replaced with an NA value.


```r
Training_Data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
Eval_Data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
```



Columns that were mostly blank did not contribute well to the prediction. We will remove those from the useful dataset and chose a dataset that only includes complete columns. 


```r
feature_set <- colnames(Training_Data[colSums(is.na(Training_Data)) == 0])[-(1:7)]
model_data <- Training_Data[feature_set]

idx <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[idx,]
testing <- model_data[-idx,]
```

We will now build 5 random forests with 150 trees each. We will make use of parallel processing to build this model.


```r
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
randomForest(x, y, ntree=ntree) 
}
```

Provide error reports for both training and test data.

```r
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  942    5    0    0
##          C    0    2  850   18    1
##          D    0    0    0  785    1
##          E    0    0    0    1  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9933          
##                  95% CI : (0.9906, 0.9954)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9942   0.9764   0.9978
## Specificity            0.9986   0.9987   0.9948   0.9998   0.9998
## Pos Pred Value         0.9964   0.9947   0.9759   0.9987   0.9989
## Neg Pred Value         1.0000   0.9982   0.9988   0.9954   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1733   0.1601   0.1833
## Detection Prevalence   0.2855   0.1931   0.1776   0.1603   0.1835
## Balanced Accuracy      0.9993   0.9957   0.9945   0.9881   0.9988
```

Conclusions and Test Data Submit
--------------------------------

As can be seen from the confusion matrix this model is very accurate. I did experiment with PCA and other models, but it was not as accurate. The test data was approximately 99% accurate. I expected nearly all of the submitted test cases to be correct which turned out to be true.

__The data for this project came from: http://groupware.les.inf.puc-rio.br/har __




```r
x <- Eval_Data
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


