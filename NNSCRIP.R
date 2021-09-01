library(ISLR)
library(tidyverse)
library(caret)
library(keras)
library(neuralnet)
library(Hmisc)
 
data = read.csv("Datos.csv") %>%
  filter(G3 != 0)

data_nn= data %>%
  dplyr::select(,c("G2","G1","age","studytime","failures",
                   "G3","traveltime","absences"))#,"Medu",
                   #"Fedu","famrel","freetime","goout"
                   #,"Dalc","Walc","health"))
test_data_nn = test_data %>%
  dplyr::select(,c("G2","G1","age","studytime","failures",
                   "traveltime","absences"))#,"Medu",
                   #"Fedu","famrel","freetime","goout"
                   #,"Dalc","Walc","health"))
  

x = data.frame(data_nn) %>%
  dplyr::select(-c("G3"))
y = data.frame(data_nn) %>%
  dplyr::select(c("G3"))

mean_x = apply(x, 2, mean)
mean_y = apply(y, 2, mean)
sd_x = apply(x, 2, sd)
sd_y = apply(y, 2, sd)

x_scaled = scale(x, center = mean_x, scale = sd_x) %>%
  data.frame()
y_scaled = scale(y, center = mean_y, scale = sd_y) %>%
  data.frame()

data_scaled = cbind(x_scaled, y_scaled)

#test_data
set.seed(123)

nn=neuralnet(G3 ~ .,data=data_scaled, hidden=10,act.fct = "logistic",
             linear.output = TRUE,stepmax=10^5,threshold = 0.01)

Predict=compute(nn,x_scaled)

pp = Predict$net.result*sd_y+mean_y


nn_vs = data.frame("Real" = y,"NN"= pp)


print(rmse2(nn_vs$G3,nn_vs$NN))
err_por(nn_vs$Real,nn_vs$NN)
plot(nn)

featurePlot(x=x,y=y, plot="box")
