setwd("~/R/PEP 2")

#Funciones
RMSE <- function(error) { sqrt(mean(error^2)) }
rmse2 = function(actual, predicted) {
  a= sqrt(mean((actual - predicted) ^ 2))
  #print(paste0("Ra�Z Error Cuadr�tico Medio: ", a))
}
err_por = function(actual, predicted) {
  a = mean((abs(actual -predicted)/actual)*100)
 # print(paste0("Error porcentual:  ", a))
}

library(ISLR)
library(tidyverse)
library(caret)
library(keras)
library(neuralnet)
library(Hmisc)
require(glmnet) 
require(lmtest)
library(rgdal)
library(raster)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrgram)
library(corrplot)
library(caTools)

# Se extraen los ceros de la variable respuesta 
data = read.csv("Datos.csv") %>%
  filter(G3 != 0)

#16 num 16 fact
#Data num�rica (286 filas)

data_number = data[,c("G3","G2","G1","Walc","goout",
                      "Medu","Fedu","famrel","Dalc",
                      "freetime","health","age","failures",
                      "absences","studytime","traveltime")]

#Data con factores 
#data_all =

#Entrenamiento y Testeo
set.seed(123)
sample <- sample.split(data_number$G3, SplitRatio = 0.7)
train <- data_number[sample, ]
test <- data_number[!sample, ]

#Correlaci�n entre variables
cor(data_number)
corrplot(cor(data_number), method="square",title="Regression between numeric columns")
corrgram(data_number, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)

# Sobre G3
# Cor + alta con G1,G2
# Un poco m�s baja con Fedu y Medu
# Cor - con failures,absences,age,Walc,goout

# A priori se esperar�a un modelo lineal :
# G3 = b + bi(g1g2fedumedu) - bj(failabsageWalcgoout) : i[1,4]j[5,9]

# Cabe mencionar que:
# 1. Cor + alta entre Fedu y Medu
# 2. Cor + entre study y G1 es mayor que 2y3
# 3. Cor + ligera entre age y fail,abs 
# 4. Cor - medalta entre study y Walc,Dalc,Freetime
# 5. Cor + alta entre Walc,gout,Dalc


# Boxplot data_num�rica

# Se extraen las ausencias para observar
# debido a que tiene muchos outliers

sin_ab= data_number %>%
  dplyr::select(,-c("absences"))
boxplot(sin_ab,
        main = "Data Sin Escalar")
boxplot(scale(sin_ab),
        main = "Data Escalada")

# Escaladas y sin escalar se ve un buen comportamiento
# en cuanto a distribuci�n de sus datos, con pocos outliers 

# Boxplot data_all

#boxplot(data_all$G3 ~ data_all$sex,
 #       xlab = "", ylab = "G3",
  #      las = 2)
#points(tapply(data_all$G3, data_all$sex, mean), pch = "x")
#abline(h = mean(data_all$G3),
 #      col = "red",
  #     lty = 4)

# Distribuci�n de cada factor de forma independiente

# En general no hay ninguno que presente un comportamiento 
# que haga pensar que pueden influir de manera significativa en G3

# De todas formas hay algunos que tienen una ligera tendencia (higher)
# y se podr�a hacer un an�lisis m�s exhaustivo sobre su significancia  
# pero por temas de tiempo no se trabajar� con factores  

# Esto no incluye los factores que est�n denotados por una escala num�rica,
# ya que esos entregan informaci�n en la correlaci�n y en boxplot 
#que s� muestran una ligera significancia o tendencia en relaci�n a G3
#(Fedu,Medu,traveltime,goout,Walc)


# Scaling Train

x = data.frame(train) %>%
  dplyr::select(-c("G3"))
y = data.frame(train) %>%
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

# Scaling Test
x_test = data.frame(test) %>%
  dplyr::select(-c("G3"))
y_test = data.frame(test) %>%
  dplyr::select(c("G3"))

mean_test_x = apply(x_test, 2, mean)
mean_test_y = apply(y_test, 2, mean)
sd_test_x = apply(x_test, 2, sd)
sd_test_y = apply(y_test, 2, sd)

x_test_scaled = scale(x_test, center = mean_test_x, scale = sd_test_x) %>%
  data.frame()
y_test_scaled = scale(y_test, center = mean_test_y, scale = sd_test_y) %>%
  data.frame()

data_test_scaled = cbind(x_test_scaled, y_test_scaled)


# Modelamiento
# Solo con variables num�ricas 

# Modelo 1: OLS

mod_1 = lm(G3 ~ .,
           data = data_scaled)

summary(mod_1)

pred.ols = predict(mod_1, newdata = x_test_scaled)*sd_test_y + mean_test_y 
p = data.frame("G3" = y_test,"OLS" = pred.ols)

# RMSE Test
ols = rmse2(p$G3,p$OLS)  

# RMSE Train
ols.tr = RMSE(mod_1$residuals) 

#Error % 
err_ols = err_por(p$G3,p$OLS)

# Residuos parecen ser homoced�sticos
plot(mod_1$residuals) 
# Pero el resto no sabr�a decir 
plot(mod_1)

# An�lisis OLS w data_number

# R2 0.9362-p mod ***-RMSE t 0.2455 - RMSE T 0.7923 - Error 6.1234%

# Para verificar su funcionamiento habr�a que validar
# los test estad�sticos que verifiqu� en otro script 
# y no los cumple

# G1yG2 muestran ***
# healthyfamrel *
# absences .

# Para mejorar habr�a que ir sacando variables y 
# analizando como cambia R2 y significancia de las otras 

# Lo cual luego de mucha prueba y error, solo con una o dos
# variables se logra significancia completa
# mas no ayuda a validar supuestos ni a predecir mejor

# Por lo que para mejorar la predicci�n, se probar� con 
# modelos de optimizaci�n iterativos


# Modelo 2 Elastic net 

grid = expand.grid(alpha = seq(from = 0.1, # 1 Lasso, 0 Ridge
                               to = 0.9,
                               length.out = 5),
                   lambda = seq(from = 0.001, # penal param
                                to = 0.1,
                                length.out = 5))

control = trainControl(search = "grid",
                       method = "cv",
                       number = 10)

set.seed(123)

(elastic_cv = train(G3 ~ .,
                    data = data_scaled,
                    method = "glmnet", 
                    trControl = control,
                    metric = "RMSE",
                    tuneGrid = grid))

elastic_cv$bestTune

elastic_model = glmnet(x = x_scaled %>%
                         as.matrix(),
                       y = y_scaled %>%
                         as.matrix(),
                       intercept = TRUE,
                       alpha = elastic_cv$bestTune$alpha,
                       lambda = elastic_cv$bestTune$lambda)

pred.elas = predict(elastic_model, newx = x_test_scaled %>% as.matrix())*sd_test_y + mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,"Elstc" = pred.elas)
colnames(p) = c("G3","OLS","Elstc")

# RMSE Test
els = rmse2(p$G3,p$Elstc)  

# RMSE Train
els.tr = min(elastic_cv$results$RMSE)

#Error % 
err_elas = err_por(p$G3,p$Elstc)

# RMSE t 0.255520 - RMSE T 0.8240 - Error 6.3621%
# Ambos son peores que los del modelo lineal 


# Modelo 3: Random Forest 

set.seed(123)

mod.rf <- train(G3 ~ ., method = "rf", data = data_scaled)
pred.rf <- predict(mod.rf, x_test_scaled)*sd_test_y + mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst")

# RMSE Train
frst.tr = min(mod.rf$results$RMSE)

# RMSE Test
frst = rmse2(p$G3,p$Rndm_Frst)

#Error % 
err_frst = err_por(p$G3,p$Rndm_Frst)


#RMSE t 0.28476 - RMSE T 0.464885 - Error 3.52 %

# Train es mayor que los anteriores, Test es mucho menor, Error es el menor


# Modelo 4: Artificial Neural Network  hid(10)thres(0.01)

set.seed(123)

nn=neuralnet(G3 ~ .,data=data_scaled, hidden=1,act.fct = "logistic",
             linear.output = TRUE,stepmax=10^5,threshold = 0.5, algorithm = 'slr')

Predict=compute(nn,x_test_scaled)
#Predict2=compute(nn,x_scaled)

pred.nn = Predict$net.result*sd_test_y+mean_test_y
#pred.nn2 = Predict$net.result*sd_y+mean_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Nrl_Ntwrk" = pred.nn)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst","Nrl_Ntwrk")

# RMSE Train
#ann.train = rmse2(train$G3,pred.nn2(1:201))

#RMSE TEST
ann = rmse2(p$G3,p$Nrl_Ntwrk)

#Error %
err_ann = err_por(p$G3,p$Nrl_Ntwrk)

# RMSE T 0,698235 - Error 5.459%
# El mejor sigue siendo el random en testeo

# Red neuronal
plot(nn)


#Correlaci�n entre modelos
cor(p)
corrplot(cor(p), method="square",title="Regression between numeric columns")
corrgram(p, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)


#Modelo Final: Ensemble 

Ensemble = p %>%
  rowwise() %>%
  mutate(Ensemble = mean(c(OLS,
                           Elstc,
                           Rndm_Frst)))  %>%
                           #,Nrl_Ntwrk))) %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble"))

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Nrl_Ntwrk" = pred.nn, "Ensemble" = Ensemble)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst","Nrl_Ntwrk","Ensemble")

# RMSE Train
nsmbl = rmse2(p$G3,p$Ensemble)

#Error %
err_nsmbl= err_por(p$G3,p$Ensemble)


#Comparaci�n de RMSE Train 

cor(p)

raices = data.frame("OLS" = ols,
                    "Elstc" = els, 
                    "Rndm_Frst" = frst,
                    "Nrl_Ntwrk" = ann, 
                    "Ensemble" = nsmbl)

#Grafico de Pred v/s G3_test
p[,-c(5)] %>%
  gather(key = "method", value = "value",
         -G3) %>%
  ggplot() +
  geom_point(aes(x = G3,
                 y = value,
                 colour = method),
             alpha = 0.8) +
  theme_classic() +
  theme(legend.position = "bottom",
        panel.border = element_rect(fill = NA, colour = "black")) +
  labs(y = "Predicted",
       colour = "Method",
       title = paste("Elstc", round(els,3),
                     ", OLS", round(ols,3),
                     ", Rndm", round(frst,3),
                     ", Ensmbl", round(nsmbl,3)))



# Finally 
# It's done 

# Dummy no empeora el modelo de forma dr�stica
# Pero se obtuvieron resultados un poco peores 
# As� q no se usan para el modelo final




compar = data.frame("RMSE" = c("Train",
                               "Test",
                               "Error %"),
                    "OLS" =c(ols.tr,ols,err_ols),
                    "Elastic" =c(els.tr,els,err_elas),
                    "Forest" =c(frst.tr,frst,err_frst),
                    "NN" =c("N/A",ann,err_ann),
                    "Ensemble" = c(mean(ols.tr,els.tr,frst.tr),
                                   nsmbl,err_nsmbl))








