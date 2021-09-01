setwd("~/R/PEP 2")


#esta es una prueba de anova con todo 
data = read.csv("Datos.csv")

summary(data)

#transforma todo a factores
data = data %>%
  mutate(sex = factor(sex),
         higher = factor(higher),
         guardian = factor(guardian),
         Pstatus = factor(Pstatus),
         Mjob = factor(Mjob),
         Fjob = factor(Fjob),
         reason = factor(reason),
         famsize = factor(famsize),
         address = factor(address),
         romantic = factor(romantic),
         higher = factor(higher),
         Medu = factor(Medu),
         Fedu = factor(Fedu),
         schoolsup = factor(schoolsup),
         famsup = factor(famsup),
         paid = factor(paid),
         activities = factor(activities),
         nursery = factor(nursery),
         internet = factor(internet),
         famrel = factor(famrel),
         freetime = factor(freetime),
         goout = factor(goout),
         Dalc = factor(Dalc),
         Walc = factor(Walc),
         health = factor(health),
         traveltime = factor(traveltime))

summary(data)

data = data %>% 
  dplyr:: select(,c("G3","Mjob"))

#es para comparar un anova q tiene todo
model_a = lm(G3 ~ .,
             data = data)

summary(model_a)
plot(model_a)

Anova(model_a,
      data = data,
      type = 2)

RMSE(model_a$residuals)

pairs(data)

#estos resultados estaban malísimos, pero aún así mejores que con pocas
ab_ic = data.frame(model_a = c(AIC = AIC(mod_1),
                               BIC = BIC(mod_1),
                               Normality = shapiro.test(mod_1$residuals)$p.value,
                               Homocedasticity = bptest(mod_1)$p.value,
                               Autocorrelation = dwtest(mod_1,
                                                        alternative = "two.sided",
                                                        iterations = 1000)$p.value,
                               Autocorrelation = bgtest(mod_1,
                                                        order = 1)$p.value,
                               Autocorrelation = bgtest(mod_1,
                                                        order = 2)$p.value))





data_all = data %>%
  mutate(sex = factor(sex),
         higher = factor(higher),
         guardian = factor(guardian),
         Pstatus = factor(Pstatus),
         Mjob = factor(Mjob),
         Fjob = factor(Fjob),
         reason = factor(reason),
         famsize = factor(famsize),
         address = factor(address),
         romantic = factor(romantic),
         higher = factor(higher),
         schoolsup = factor(schoolsup),
         famsup = factor(famsup),
         paid = factor(paid),
         activities = factor(activities),
         nursery = factor(nursery),
         internet = factor(internet))
summary(data_all)