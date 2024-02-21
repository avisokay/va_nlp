library(Tariff)

data("RandomVA3") 
test<-RandomVA3[1:200,] 
train<-RandomVA3[201:400,] 
allcauses<-unique(train$cause) 
fit <-tariff(causes.train="cause",symps.train=train, symps.test=test, causes.table =allcauses)
plot(fit,top=10,main="Top 5 PopulationCODdistribution") 
plot(fit,min.prob=0.05,main ="PopulationCODdistribution(atleast5%)")

fit <-tariff(causes.train="cause",symps.train=train,
             symps.test=test,causes.table =allcauses) 
correct<-which(fit$causes.test[,2]==test$cause) 
accuracy<-length(correct)/dim(test)[1] 
summary(fit) 
summary(fit,top=10) 
summary(fit,id="p849",top= 3)









