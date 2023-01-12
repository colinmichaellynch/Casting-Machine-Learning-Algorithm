rm(list = ls())

library(ggplot2)
library(wesanderson)
library(RColorBrewer)
library(plotrix)

setwd("C:/Users/user/Documents/ants_as_ensembles/UpdatedCode")
data = read.csv('IrisAntClassification.csv')
data$sumError = data$Bias+data$Variance

data2 = read.csv('RFandAdaboost.csv')
data2Adaboost = subset(data2, Model == "Adaboost")
data2RF = subset(data2, Model == "RandomForest")
dataAdaboost = subset(data, P ==1)
dataRF = subset(data, P ==min(data$P))

windowSizes = unique(data$WindowSize)
PVec = c()
for(i in 1:nrow(data)){
  
  PVec[i] = sum(data$WindowSize[i]>=windowSizes)/length(windowSizes)
  
}
data$P= PVec

ggplot(data, aes(x = P, y = ValidationAccuracy)) + geom_point(size = 2) + geom_smooth() + theme_bw() + labs(x = "P", y = "Accuracy") + theme(text = element_text(size=14)) + geom_point(data = data2Adaboost, aes(x = 1.01, y = ValidationAccuracy), color = "red", size = 2)  + geom_point(data = data2RF, aes(x = .1, y = ValidationAccuracy), color = "blue", size = 2) 

ggplot(data, aes(x = P, y = Bias)) + geom_point(size = 2) + geom_smooth() + theme_bw() + labs(x = "P", y = "Bias") + theme(text = element_text(size=14)) + geom_point(data = data2Adaboost, aes(x = 1.01, y = Bias), color = "red", size = 2) + geom_point(data = data2RF, aes(x = .1, y = Bias), color = "blue", size = 2) 

ggplot(data, aes(x = P, y = Variance)) + geom_point(size = 2) + geom_smooth() + theme_bw() + labs(x = "P", y = "Variance") + theme(text = element_text(size=14)) + geom_point(data = data2Adaboost, aes(x = 1.01, y = Variance), color = "red", size = 2) + geom_point(data = data2RF, aes(x = .1, y = Variance), color = "blue", size = 2) 

ggplot(data, aes(x = P, y = Time)) + geom_point(size = 2) + geom_smooth() + theme_bw() + labs(x = "P", y = "Time Elapsed (s)") + theme(text = element_text(size=14)) + geom_point(data = data2Adaboost, aes(x = 1.01, y = Time), color = "red", size = 2) + geom_point(data = data2RF, aes(x = .1, y = Time), color = "blue", size = 2) 

#bias vs variance tradeoff
dataSummary = aggregate(c(data$Bias), by = list(as.character(round(data$P, digits = 1))), FUN = function(x) mean = mean(x))
dataSummary2 = aggregate(c(data$Bias), by = list(as.character(round(data$P, digits = 1))), FUN = function(x) se = std.error(x))
dataSummary3 = aggregate(c(data$Variance), by = list(as.character(round(data$P, digits = 1))), FUN = function(x) mean = mean(x))
dataSummary4 = aggregate(c(data$Variance), by = list(as.character(round(data$P, digits = 1))), FUN = function(x) se = std.error(x))
dataSummary$biasse = dataSummary2$x
dataSummary$varmean = dataSummary3$x
dataSummary$varse = dataSummary4$x
colnames(dataSummary) = c("P", "MeanBias", "SEBias", "MeanVar", "SEVar")

ggplot(dataSummary, aes(x = MeanBias, y = MeanVar, color = P)) + scale_color_brewer(palette="Spectral") + theme_bw() + labs(color='P') + theme(text = element_text(size = 14)) + ylab("Mean Variance") + xlab("Mean Bias")+ theme(aspect.ratio=1) + geom_point(size = 5)

#+geom_pointrange(aes(ymin=MeanVar-SEVar, ymax=MeanVar+SEVar), size = .75)+geom_pointrange(aes(xmin=MeanBias-SEBias, xmax=MeanBias+SEBias), size = .75)

wilcox.test(data2Adaboost$ValidationAccuracy, dataAdaboost$ValidationAccuracy, alternative = "two.sided")
wilcox.test(data2Adaboost$Bias, dataAdaboost$Bias, alternative = "two.sided")
wilcox.test(data2Adaboost$Variance, dataAdaboost$Variance, alternative = "two.sided")
wilcox.test(data2Adaboost$Time, dataAdaboost$Time, alternative = "two.sided")

wilcox.test(data2RF$ValidationAccuracy, dataRF$ValidationAccuracy, alternative = "two.sided")
wilcox.test(data2RF$Bias, dataRF$Bias, alternative = "two.sided")
wilcox.test(data2RF$Variance, dataRF$Variance, alternative = "two.sided")
wilcox.test(data2RF$Time, dataRF$Time, alternative = "two.sided")

#Find P where accuracy is max
#Find window size where bias and variance are minimized
#See which free parameter has a stronger effect, lr or ws 
#Find which has shortest elapsed time 
#fix w = 1 problem 

ggplot(data, aes(x = LearningRate, y = ValidationAccuracy)) + geom_point() + geom_smooth()
ggplot(data, aes(x = WindowSize, y = ValidationAccuracy)) + geom_point() + geom_smooth()

ggplot(data, aes(LearningRate, P, z = ValidationAccuracy))+ geom_contour_filled() + labs(x = "Learning Rate", y = "P", fill = "Validation Accuracy") + theme_bw()  + theme(text = element_text(size=14))

lrData = subset(data, LearningRate == .1)

ggplot(lrData, aes(x = P, y = ValidationAccuracy)) + geom_point(size = 2) + geom_smooth() + theme_bw() + labs(x = "P", y = "Validation Accuracy") + theme(text = element_text(size=14)) 
