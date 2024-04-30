library(dplyr) #Package to clean and format the data 
library(ggplot2) #Package to plot data
library(psych) #Package for corplot
library(FactoMineR)#Package  for Factor analysis
library(factoextra)#Package  for Factor analysis
library(caret)#Package for var. importance
library(rpart) #Decision Tree
library(naivebayes)#Naive Bayes model
library(pROC) #ROC 


##Load Data from CSV
df<-read.csv('/Users/nonavance/Desktop/SNHU/Capstone in Data Analytics /Final Project/EmployeeAttrition_Data.csv')
str(df) #checking data types
summary(df) #summery statistics
sum(is.na(df)) #checking for missing data
length(unique(df$EmployeeNumber)) == nrow(df) #checking for duplicates

dim(df) #retrieve the dimensions of dataframe
##Pipe to clean and format data##
columns_to_convert <- c("Attrition","BusinessTravel","Department","Education","EducationField",
                        "EnvironmentSatisfaction","Gender" ,"JobInvolvement","JobLevel", "JobSatisfaction", 
                        "JobRole", "MaritalStatus" ,"OverTime" , "PerformanceRating", 
                        "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance")

df<-df %>% select(c(-EmployeeNumber,-Over18, -StandardHours, -EmployeeCount)) %>% #dropping columns 
                  mutate_at(vars(columns_to_convert ), factor) #converting columns to factor 
  
##Limitations of the pipe cleaning 
## if new data has different column names the pipe will not run 

#-------------------------------------------------Data Exploration-------------------------------------------------------------# 
str(df) #internal structure of dataframe
dev.off() #close graphical devices that were opened during a session. 
          #Run this if the graph does not display
   
#Attrition Bar Plot
ggplot(df, aes(x=Attrition,fill=Attrition))+
       geom_bar(stat="count", width=0.7)+
       geom_text(stat='count', aes(label=..count..), vjust=0.05)+
       labs(title = "bar plot",x="Attrition", y="Count")+theme_bw()+
       labs(title = "Attrition Count", subtitle =" ",caption = '', 
       x="Attrition No/Yes", y="Count")+theme_bw()+
       scale_fill_manual(values = c("blue3", "orange"))

#Correlation Plot
corPlot(select_if(df, is.numeric),cex = 1.25,upper = FALSE)

#Monthly Income histogram
ggplot(df, aes( MonthlyIncome))+ geom_histogram()+ 
       geom_histogram(aes(fill = Attrition),color = "black", alpha = 0.7) +
       ggtitle("Monthly Income Distribution")+
       scale_fill_manual(values = c("blue3", "orange"))+ theme_minimal()

#Hypothesis Testing Monthly Income
#There is a difference in the mean Monthly Income among groups of people:
#Group of people who leave the company.
#Group of people who stay in the company.
t.test(select(filter(df, Attrition == "No"), MonthlyIncome), 
       select(filter(df, Attrition == "Yes"), MonthlyIncome))

#Age Histogram
ggplot(df, aes(Age))+ 
       geom_histogram(aes(fill = Attrition),color = "black", alpha = 0.7, bins=15) +
       ggtitle("Age Distribution")+scale_fill_manual(values = c("blue3", "orange"))+ theme_minimal()

#Hypothesis Testing Age
#There is a difference in the mean of Age among groups of people:
#Group of people who leave the company.
#Group of people who stay in the company
t.test(select(filter(df, Attrition == "No"), Age), 
       select(filter(df, Attrition == "Yes"), Age))

#JobeRole bar Plot
ggplot(data=df, aes(x=JobRole, fill=Attrition))+
       geom_bar(stat = "count", position = "fill")+ggtitle("Jobe Role")+
       scale_fill_manual(values = c("blue3", "orange"))+ theme_minimal()

#EnvironmentSatisfaction bar Plot
ggplot(data=df, aes(x=EnvironmentSatisfaction, fill=Attrition))+
       geom_bar(stat = "count", position = "fill")+
       ggtitle("Environment Satisfaction")+
       scale_fill_manual(values = c("blue3", "orange"))+ theme_minimal()


#-----------------------------------------------------Factor Analysis-----------------------------------------------------------------#

res.famd <- FAMD(df, 
                 sup.var = 2,  ## Set the target variable "Attrition" 
                 ##as a supplementary variable, so it is not included in the analysis for now
                 graph = FALSE, 
                 ncp=25)

# Inspect principal components
get_eigenvalue(res.famd)


#Contribution of variables Dim-1
fviz_contrib(res.famd,
             choice = "var",
             axes = 1,
             top = 10, color = 'darkorange3', barfill  = 'blue3',fill ='blue3')

#Contribution of variables Dim-2
fviz_contrib(res.famd,
             choice = "var",
             axes = 2,
             top = 10, color = 'darkorange3', barfill  = 'blue3',fill ='blue3')

#the variables and the curve that help you choose the number of dimension
fviz_screeplot(res.famd, ncp=14,linecolor = 'darkorange3', barfill  = 'blue3', 
               addlabels = TRUE,
               barcolor ='blue3', xlab = "Dimensioni", 
               ylab = '% varicance',
               main = 'Reduction of components')

# Use habillage to specify groups for coloring
fviz_pca_ind(res.famd,
             label = "none", # hide individual labels
             habillage = df$Attrition, # color by groups
             palette = c("#00AFBB", "#FC4E07"),
             addEllipses = TRUE # Concentration ellipses
)

#-------------------------------------------Variable Importance in RPart Model Classification Trees----------------------------------------#

set.seed(100) #seed for reproducibility 

rPartMod <- rpart(Attrition ~ ., data=df,  method="class") #running RPart Model

V <- varImp(rPartMod, permut = 1) #Varibel importance 

#Plotting varibel importance 
ggplot2::ggplot(V, aes(x=reorder(rownames(V),Overall), y=Overall)) +
                geom_point( color="blue", size=4, alpha=0.6)+
                geom_segment( aes(x=rownames(V), xend=rownames(V), y=0, yend=Overall), 
                color='skyblue') +
                xlab('Variable')+
                ylab('Overall Importance')+
                theme_light() +
                coord_flip() 
#---------------------------------------------UpSampling Data---------------------------------------------------------#
set.seed(123) #seed for reproducibility 
train_up<-upSample(df, df$Attrition) #upsampling the train data set

train_up<- train_up[, -2] %>% #removing old Attrition column
           rename(Attrition = names(.)[31]) %>% #renaming new column
           mutate(Attrition = factor(Attrition)) #converting to factor

#plot train_up
ggplot(train_up, aes(x=Attrition,fill=Attrition))+
       geom_bar(stat="count", width=0.7)+
       geom_text(stat='count', aes(label=..count..), vjust=0.05)+
       labs(title = "bar plot",x="Attrition", y="Count")+theme_bw()+
       labs(title = "Attrition Count", subtitle =" ",caption = '', 
       x="Attrition No/Yes", y="Count")+theme_bw()+
       scale_fill_manual(values = c("blue3", "orange"))

#------------------------------------------------Naive Bayes Model-----------------------------------------------------------#

#creating/training the model 
nb_model <- naive_bayes(Attrition ~ Age+BusinessTravel+Department+DistanceFromHome+EnvironmentSatisfaction+
                                   HourlyRate+JobRole+JobSatisfaction+MonthlyIncome+OverTime+PercentSalaryHike+
                                   WorkLifeBalance+YearsSinceLastPromotion+JobInvolvement+StockOptionLevel,
                                   data = train_up,laplace = 1)

#predicting on the traing set 
prediction <- predict(nb_model, 
              newdata = select(train_up,Age,BusinessTravel,Department,DistanceFromHome,EnvironmentSatisfaction,
                              HourlyRate,JobRole,JobSatisfaction,MonthlyIncome,OverTime,PercentSalaryHike,
                              WorkLifeBalance,YearsSinceLastPromotion,JobInvolvement,StockOptionLevel))

#ConfusionMatrix training set
confusionMatrix(factor(prediction), train_up$Attrition, positive = "Yes")

#-----------------------------------------Pipeline Prediction new data--------------------------------------------#

#Load new data
df_test<-read.csv('/Users/nonavance/Desktop/SNHU/Capstone in Data Analytics /Final Project/EmployeeAttrition_Verify.csv')

#pip for data transfr
df_test<-df_test %>% select(c(-EmployeeNumber,-Over18, -StandardHours, -EmployeeCount)) %>% #dropping columns 
         mutate_at(vars(columns_to_convert ), factor)#converting columns to factor 

#predict NB model test data
prediction_test <- predict(nb_model,
                   newdata = select(df_test,Age,BusinessTravel,
                                   Department,DistanceFromHome,EnvironmentSatisfaction,
                                   HourlyRate,JobRole,JobSatisfaction,MonthlyIncome,
                                   OverTime,PercentSalaryHike,WorkLifeBalance,YearsSinceLastPromotion,
                                   JobInvolvement,StockOptionLevel))


#Conf. Matrix test data
confusionMatrix(as.factor(prediction_test), df_test$Attrition, positive = "Yes")

#------------------------------------------------ROC---------------------------------------------------------------#
#ROC curve 
par(pty='m')
#converting "Yes' to 1 and "No" to 0 
obsr<-ifelse(df_test$Attrition=="Yes",1,0)
pred<-ifelse(prediction_test=='Yes', 1, 0)

#predicting on test data probabilities 
prediction_test_prob <- data.frame(predict(nb_model,
                           newdata = select(df_test,Age,BusinessTravel,
                                            Department,DistanceFromHome,EnvironmentSatisfaction,
                                            HourlyRate,JobRole,JobSatisfaction,MonthlyIncome,
                                            OverTime,PercentSalaryHike,WorkLifeBalance,YearsSinceLastPromotion,
                                            JobInvolvement,StockOptionLevel), type='prob'))

#Plotting ROC 
prediction_test_prob$Yes
pred <- prediction(prediction_test_prob$Yes, obsr)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE, lwd = 3)

#AUC 
auc(roc( obsr, pred))

#----------------------------------Data_Drift_Test-----------------------------------------------#
#perform Kolmogorov-Smirnov test for numeric variabels example using Age
#Null Hypothesis two sample datasets come from the same distribution
#Alternative two sample datasets come from the different distribution
ks.test(df$Age, df_test$Age)

#chi-squared test for two samples,
#Null hypothesis there is no significant difference between the observed and expected frequencies in the two samples.
tab1<-table(df$Gender,df$Attrition)#contingency table sample 1
tab2<-table(df_test$Gender,df_test$Attrition)#contingency table sample 2
tab <- rbind(tab1, tab2) #combine tables
result <- chisq.test(tab)# chi test



