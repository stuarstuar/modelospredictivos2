setwd("~/R/PEP 2")

data = read.csv("Datos.csv")
test_data = read.csv(file = "Test.csv") #se usa cuando tenemos modelo con predict

#library
require(tidyverse)
require(caret)
require(glmnet)
require(lmtest) #homocedasticity test
require(car)
require(MASS)
require(emmeans) #tukey test (not used)


test_data = test_data %>%
 # mutate(higher = factor(higher))
  dplyr::select(c("age","studytime","failures",
                  "absences","G2","G1"))

summary(test_data)




#variables luego del descarte del boxplot 
#(por ahora solo una idea)
variables = c("age" ,"studytime","failures",
             "absences","higher", "G1", "G2", "G3")

#variables q igual tenían tendencia
#variables2 = c("Fedu","Medu","internet"
#              ,"Mjob","Fjob","traveltime",
#              "schoolsup")

var_exp_1 = c("age" ,"studytime","failures",
            "absences","G1")
var_exp_2 = c("age" ,"studytime","failures",
            "absences","higher","G2")

data0 = data[,variables] #incluye g3 
data1 = data[,var_exp_1] #usados para explicar g1
#data2 = data[,var_exp_2] #usados para explicar g2 (en vd intentare con g1 y si resulta, lo aplico)

#data1 = data1 %>%
 # mutate(#higher = factor(higher),
  #       failures = factor(failures))

summary(data1)


#modelo 1 tiene una regresion para G1 con los datos de var_exp_1
modelo_1 = lm(G1 ~ .,
              data = data1)

png("residuos.png", width = 1200, height = 650)
par(mfrow = c(2, 2), cex = 1.5)
plot(modelo_1)


dev.off()

Anova(modelo_1,
      data = data1,
      type = 3)

modelo_step = step(modelo_1,
                   direction = "both")

summary(modelo_1)
summary(modelo_step)

plot(modelo_1)


ab_ic = data.frame(model_1 = c(AIC = AIC(modelo_1),
                               BIC = BIC(modelo_1),
                               Normality = shapiro.test(modelo_1$residuals)$p.value,
                               Homocedasticity = bptest(modelo_1)$p.value,
                               Autocorrelation = dwtest(modelo_1,
                                                        alternative = "two.sided",
                                                        iterations = 1000)$p.value,
                               Autocorrelation = bgtest(modelo_1,
                                                        order = 1)$p.value,
                               Autocorrelation = bgtest(modelo_1,
                                                        order = 2)$p.value),
                   model_step = c(AIC = AIC(modelo_step),
                                  BIC = BIC(modelo_step),
                                  Normality = shapiro.test(modelo_step$residuals)$p.value,
                                  Homocedasticity = bptest(modelo_step)$p.value,
                                  Autocorrelation = dwtest(modelo_step,
                                                           alternative = "two.sided",
                                                           iterations = 1000)$p.value,
                                  Autocorrelation = bgtest(modelo_step,
                                                           order = 1)$p.value,
                                  Autocorrelation = bgtest(modelo_step,
                                                           order = 2)$p.value))


# variance inflation factor
vif(modelo_1)

boxplot(data1)
#data1 = scale(data1) %>%
 # data.frame()
#summary(data1)



#de saquí se compara el test en g1 con el modelolineal 
modelo_lm_tst_pred = predict(modelo_1, newdata = test_data) %>%
  data_frame()

colnames(pred_lm_vs_test_g1) = c("Linear", "Test")

pred_lm_vs_test_g1 =  data.frame("Modelo" = modelo_lm_tst_pred, "Test" = test_data$G1)

pred_lm_vs_test_g1 %>%
  ggplot() +
  geom_point(aes(x = .,
                y = Test))+
  geom_line(aes(x= .,
                y= Test),
            colour ="red")

pairs(data0)

summary(data0)





data0 = scale(data0) %>%
  data.frame()

pairs(data0)














