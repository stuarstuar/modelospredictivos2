setwd("~/R/PEP 2")

require(tidyverse)
require(caret)
require(glmnet) 
require(kknn) 
require(lmtest)
library(rgdal)
library(raster)

data = read.csv("Datos.csv") %>%
  filter(G3 != 0)

data_1= data %>%
  dplyr::select(,c("G2","G1","age","studytime","failures",
                   "G3","traveltime","absences"))

#Tranformación de variables
x = data.frame(data_1) %>%
  dplyr::select(-c("G3"))
y = data.frame(data_1) %>%
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

pairs(data_1)

#Modelo 1: Lineal

mod_1 = lm(G3 ~ .+G1:G2+failures:absences+traveltime:studytime,
           data = data_scaled)

RMSE <- function(error) { sqrt(mean(error^2)) }
RMSE(mod_1$residuals)

pred.ols = predict(mod_1, newdata = x_scaled)*sd_y + mean_y

plot(mod_1$residuals)
boxplot(data_scaled)
summary(mod_1)

plot(mod_1)

#Modelo 2: Elastic net

grid = expand.grid(alpha = seq(from = 0.1, # 1 Lasso, 0 Ridge
                               to = 0.9,
                               length.out = 5),
                   lambda = seq(from = 0.001, # penal param
                                to = 0.1,
                                length.out = 5))

control = trainControl(search = "grid",
                       method = "cv",
                       number = 10)

set.seed(849)

(elastic_cv = train(G3 ~ .+G1:G2+failures:absences+traveltime:studytime,
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

min(elastic_cv$results$RMSE)
pred.elas = predict(elastic_model, newx = x_scaled %>% as.matrix())*sd_y + mean_y


#Modelo 3: KNN

summary(data_scaled)

grid_2 = expand.grid(kmax = seq.int(from = 1, to = 51, by = 2), # Number of neighbors considered
                   distance = seq(from = 0.1, # Parameter of Minkowski distance
                                  to = 2,
                                  by = 0.1),
                   kernel = c("gaussian", "epanechnikov"))

control_2= trainControl(search = "grid",
                       method = "cv",
                       number = 5)
set.seed(849)

(knn_cv = train(G3 ~ .,
                data = data_scaled,
                method = "kknn", 
                trControl = control_2,
                metric = "RMSE",
                tuneGrid = grid_2))

knn_cv$bestTune

kknn_model = kknn(G3 ~ .,
                  train = data_scaled,
                  test = data_scaled,
                  k = knn_cv$bestTune$kmax,
                  distance = knn_cv$bestTune$distance,
                  kernel = as.character(knn_cv$bestTune$kernel))

pred.knn = predict(kknn_model)*sd_y + mean_y


#Modelo 4: Random Forest RF

set.seed(123)

mod.rf <- train(G3 ~ .+G1:G2+failures:absences+traveltime:studytime, method = "rf", data = data_scaled)
pred.rf <- predict(mod.rf, x_scaled)*sd_y + mean_y
min(mod.rf$results$RMSE)

# Modelo 5: Support Vector Machines SVM

set.seed(123)
mod.svm = train(G3 ~ ., method = "svmRadial", data = data_scaled)
pred.svm <- predict(mod.svm, x_scaled)*sd_y + mean_y
min(mod.svm$results$RMSE)

#Modelo 6: Ensemble

Ensemble = preds %>%
  rowwise() %>%
  mutate(Ensemble = mean(c(OLS,
                           Elstc,
                          Rndm_Frst))) %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble"))

preds = preds %>%
  dplyr::select(,)

#OLS vs Elastic vs KKNN VS RF vs SVM vs Ensemble 

preds = data.frame(y,
                   pred.ols,
                   pred.elas
                  ,pred.rf)
                # ,Ensemble)
                 #,predict(kknn_model)*sd_y + mean_y)

colnames(preds) = c("Real", "OLS", "Elstc","Rndm_Frst")#,"Ensemble")#,"KKNN")

rmse2(preds$Real,preds$Rndm_Frst)
err_por(preds$Real,preds$Rndm_Frst)

cor(preds)

preds%>%
  gather(key = "method", value = "value",
         -Real) %>%
  ggplot() +
  geom_point(aes(x = Real,
                 y = value,
                 colour = method),
             alpha = 0.8) +
  theme_classic() +
  theme(legend.position = "bottom",
        panel.border = element_rect(fill = NA, colour = "black")) +
  labs(y = "Predicted",
       colour = "Method",
       title = paste("Elastic", round(min(elastic_cv$results$RMSE),2),
                      ", OLS", round(RMSE(mod_1$residuals),2),
                     ", Rndm", round(min(mod.rf$results$RMSE),2)))


#Ensemble con los testeos (se tiene que ejecutar sin esto antes)
Ensemble_Test= preds_test%>%
  rowwise() %>%
  mutate(Ensemble_Test= mean(c(OLS,
                               Elstc,
                               Rndm_Frst))) %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble_Test"))


#Test
test_data = read.csv("Test.csv")

test_data = test_data %>%
  dplyr::select(,c("G1","G2","age","failures",
                   "studytime","traveltime","absences"))

#Escalando data de entrenamiento
x_test = data.frame(test_data)
mean_x_test = apply(x_test, 2, mean)
sd_x_test = apply(x_test, 2, sd)
x_test_scaled = scale(x_test, center = mean_x_test, scale = sd_x_test) %>%
  data.frame()

y_G3 = y[c(207:285),]
preds_test = data.frame(y_G3
                        ,predict(mod_1, newdata = x_test_scaled)*sd_y + mean_y
                        ,predict(elastic_model, newx = x_test_scaled %>% as.matrix())*sd_y + mean_y
                        ,pred.rf <- predict(mod.rf, x_test_scaled)*sd_y + mean_y
                        ,Ensemble_Test)
                       
colnames(preds_test) = c("G3_train","OLS", "Elstc","Rndm_Frst","Ensemble_Test")


#ERROR ENTRENAMIENTO CON PREDICCION EN TEST
rmse2 = function(actual, predicted) {
  a= sqrt(mean((actual - predicted) ^ 2))
  print(paste0("RaíZ Error Cuadrático Medio: ", a))
}
err_por = function(actual, predicted) {
  a = mean((abs(actual -predicted)/actual)*100)
  print(paste0("Error porcentual:  ", a))
}

#Raiz del error cuadrado medio test
rmse2(preds_test$G3_train,preds_test$Ensemble_Test)
#Error porcentual test
err_por(preds_test$G3_train,preds_test$Ensemble_Test)

#Grafico de prediccion de testeo vs G3train
preds_test %>%
  gather(key = "method", value = "value",
         -G3_train) %>%
  ggplot() +
  geom_point(aes(x = G3_train,
                 y = value,
                 colour = method),
             alpha = 0.8) +
  theme_classic() +
  theme(legend.position = "bottom",
        panel.border = element_rect(fill = NA, colour = "black")) +
  labs(y = "Predicted",
       colour = "Method",
       title = paste("Elastic", round(min(elastic_cv$results$RMSE),2),
                     ", OLS", round(RMSE(mod_1$residuals),2),
                     ", Rndm", round(min(mod.rf$results$RMSE),2)))


# Knn se cambia al hacer el modelo 
# no se uso pq tardaba mucho y daba feo 
# Modelo Stochastic Gradient Boosting 
predDF <- data.frame(pred.rf, pred.svm, class = testing$class)
predDF_bc <- undersample_ds(predDF, "class", nsamples_class)
set.seed(123)
combModFit.gbm <- train(as.factor(class) ~ ., method = "gbm", data = predDF_bc, distribution = "multinomial")
combPred.gbm <- predict(combModFit.gbm, predDF)

