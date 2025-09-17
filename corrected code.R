library(R2jags)
library(tidyverse)
library(lubridate)
library(sf)
library(ggplot2)
library(ggpubr)
library(dplyr)

# ------- Reading and preparing data -------
all_data = readRDS('mortality/dataset.rds')

# Creating the complete space time dataframe
month = unique(all_data$month)
county = unique(all_data$county)
space_time_df = crossing(month,county)
space_time_df$id.month = as.numeric(as.factor(space_time_df$month))
space_time_df$id.county = as.numeric(as.factor(space_time_df$county))

df_jags_0 = left_join(all_data, space_time_df)

df_jags =  df_jags_0 %>% 
  mutate(monthly_cases = as.integer(monthly_cases),
         # Replacing the number of covid cases for the first 7 months of 2020 with NA
         edited_cases = ifelse(month < '2020-08-01' & month > '2019-12-01', NA, monthly_cases)) 


miss_row = which(is.na(df_jags$edited_cases)) # Missing values indicators to be used within JAGS


# ------- Modelling using JAGS -------
model_code <- "
model
{
  # Likelihood

  for (i in 1:N){
      y[i] ~ dpois(lambda[i])
      
      # Predictive & Counterfactual 
      y_pred[i] ~ dpois(lambda[i])
      y_counter[i] ~ dpois(lambda_counterfactual[i])
      
      
      log(lambda[i]) <- alpha[county[i]] + 
                        beta_str[county[i]] * str[i] +
                        beta_vac[county[i]] * vaccine[i] * cases[i] +
                        beta_temp[county[i]] * temper[i] +
                        beta_case[county[i]] * cases[i] +
                        beta_month[county[i]] * month[i]
      
      
      log(lambda_counterfactual[i]) <- alpha[county[i]] + 
                                         beta_temp[county[i]] * temper[i] +
                                         beta_case[county[i]] * cases[i] +
                                         beta_month[county[i]] * month[i]
  }
    
    # Priors for the missing x values
    for(k in 1:N_miss_x) {
      cases[miss_row[k]] ~ dpois(10*org_cases[miss_row[k]])
    }

  # Priors
  for (j in 1:N_county){
  
    alpha[j] ~ dnorm(mu_alpha,sigma_alpha^-2) #dnorm(0,10^-2)
    beta_str[j] ~ dnorm(mu_str, sigma_str^-2)
    beta_vac[j] ~ dnorm(mu_vac, sigma_vac^-2)
    beta_case[j] ~ dnorm(mu_case, sigma_case^-2)
    beta_month[j] ~ dnorm(mu_month, sigma_month^-2)
    beta_temp[j] ~ dnorm(mu_temp, sigma_temp^-2)


  }

  # Hyperpriors
  
  mu_str ~ dnorm(0.0, 0.01)
  mu_vac ~ dnorm(0.0, 0.01)
  mu_case ~ dnorm(0.0, 0.01)
  mu_month ~ dnorm(0.0, 0.01)
  mu_alpha ~ dnorm(0.0, 0.01)
  mu_temp ~ dnorm(0.0, 0.01)

  sigma_str ~ dt(0, 1, 1)T(0,)
  sigma_vac ~ dt(0, 1, 1)T(0,)
  sigma_case ~ dt(0, 1, 1)T(0,)
  sigma_month ~ dt(0, 1, 1)T(0,)
  sigma_alpha ~ dt(0, 1, 1)T(0,)
  sigma_temp ~ dt(0, 1, 1)T(0,)
}
"

# Set up the data
model_data <- list(
  y = df_jags$notices,
  N = nrow(df_jags),
  N_county = max(df_jags$id.county),
  county = df_jags$id.county,
  month = df_jags$id.month,
  # vars 
  str = df_jags$stringency,
  vaccine = df_jags$vaccination_rate,
  cases = df_jags$edited_cases, 
  org_cases = df_jags$monthly_cases,
  N_miss_x = length(miss_row),
  miss_row = miss_row,
  temper = df_jags$avg_temp
)

# Choose the parameters to watch
model_parameters <- c("alpha", 'beta_str','beta_vac','beta_case', 'beta_month', 'beta_temp', 
                      'y_pred','y_counter',
                      'mu_str', 'mu_vac', 'mu_case','mu_month',
                      'sigma_str', 'sigma_vac', 'sigma_case','sigma_month')

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 7000,
  n.burnin = 500,
  n.thin = 4
)

plot(model_run) # Checking the R-hat values
posteriors = model_run$BUGSoutput$sims.list # Extracting the posterior samples

# -------- Plotting Theme ----------------------------------
Theme = theme(
  panel.background = element_rect(fill = "transparent", color = "transparent"),
  plot.background = element_rect(fill = "white"),
  axis.text.x = element_text(color = "grey10",
                             size = 10, face = "bold", margin = margin(t = 6)),
  axis.text.y = element_text(color = "grey10",
                             size = 10, face = "bold", margin = margin(t = 6)),
  panel.grid.major.x = element_line(color = "grey10", linetype = "13", size = .4),
  plot.margin = margin(15, 30, 10, 30),
  plot.caption = element_text( color = "grey50", size = 8,
                               hjust = .5, margin = margin(t = 30, b = 0)),
  legend.text = element_text(color = "grey10", size = 10, face = "bold"),
  legend.title = element_blank(),   
  axis.title.x = element_text(color = "grey10", size = 10, face = "bold"),
  axis.title.y = element_text(color = "grey10", size = 10, face = "bold", angle = 90,
                              margin = margin(t = 0, r = 20, b = 0, l = 0)))

# ------ Prediction performance -------
preds <- model_run$BUGSoutput$mean$y_pred
lm_results = summary(lm(df_jags$notices ~ preds))
R_2 = trunc(lm_results$adj.r.squared*10^2)/10^2

# ---- utility function----------
get_effect = function(posteriors, effect){
  # It returns the median and the 95% credible interval of an effect 
  # from the posterior object (model_run$BUGSoutput$sims.list) of a JAGS fit
  
  #-Args: 
  #     posteriors (list): list containing model parameters posterior samples
  #     effect: (string): name of the effect (parameter)
  #-Returns:
  #     df_beta (Dataframe): a dataframe with three columns lower, middle and higher
  
  beta_lower = apply(posteriors[[effect]],2,function(x) quantile(x, probs = 0.025))
  beta_med = apply(posteriors[[effect]],2,function(x) quantile(x, probs = 0.5))
  beta_higher = apply(posteriors[[effect]],2,function(x) quantile(x, probs = 0.975))
  
  df_beta = data.frame(lower = beta_lower, middle = beta_med, higher = beta_higher)
  return(df_beta)
}

# ----- Posterior predictive performance --------
df_preds = get_effect(posteriors, 'y_pred')

df_preds = df_preds %>% 
  mutate(obs = df_jags$notices,
         within_interval = ifelse(obs > lower & obs < higher, 1 , 0))

round(mean(df_preds$within_interval),2) # Percentage of observations falling within the 95% CI

up_lim = max(df_preds[,c("obs", "middle")], na.rm = T) 
low_lim = min(df_preds[,c("obs", "middle")], na.rm = T) 

df_preds %>% 
  ggplot(aes(obs, middle)) +
  geom_point() +
  geom_linerange(aes(ymin=lower, ymax=higher),  alpha = 0.3, color = '#112f5f') + 
  geom_abline(slope=1, color = "darkblue") +
  coord_cartesian(ylim = c(low_lim,up_lim), xlim = c(low_lim,up_lim)) + 
  annotate("text", x = 150, y = 1050, label = bquote(R^2 == .(R_2))) + #
  labs(x = 'Observed', y = 'Fitted') + 
  theme_bw() +
  Theme



# ------- Plotting counterfactual mortality -------
counter <- round(model_run$BUGSoutput$mean$y_counter)
preds <- round(model_run$BUGSoutput$mean$y_pred)

df_jags_long = df_jags %>% 
  select(month, notices) %>% 
  mutate(Factual = preds,
         Counterfactual = counter) %>% 
  group_by(month) %>% 
  summarise(Factual = sum(Factual),
            Counterfactual = sum(Counterfactual)) %>% 
  ungroup() %>% 
  pivot_longer(names_to = 'Type', values_to = 'notices', -month) %>% 
  filter(year(month) > 2019)

df_ribbon <-
  df_jags_long %>% 
  group_by(month) %>% 
  summarize(
    min = min(notices),
    max = max(notices)
  ) 

ggplot(df_jags_long, aes(month, notices, color = Type)) + 
  geom_ribbon(
    data = df_ribbon,
    aes(x = month, ymin = min, ymax = max),
    alpha = .2, inherit.aes = FALSE, fill = 'red'
  )  +
  geom_line(data = filter(df_jags_long, Type == "Counterfactual"), size = 2)+
  geom_line(size = 1.5) +
  theme_bw() +
  labs(x = 'Time', y = "Number of deaths") +
  Theme

# Calculating the total number of lives saved by vaccination (covid related) and lockdowns (non-covid related)
sum(counter - df_jags$notices) 
df_counters = t(posteriors[['y_counter']])
df_counters = apply(df_counters, 2, sum) - sum(df_jags$notices) # Saved lives as the number of counterfactual deaths minus the factual deaths
quantile(df_counters, probs = c(0.025, 0.5, 0.975))



# ------- Preparing the map --------
# Read GADM v4.1 level-1 layer for Ireland
# (works with a folder + layer OR the .shp file path)
# e.g., if you unzipped .../gadm41_IRL_shp/gadm41_IRL_1.shp
map <- st_read(dsn = "mortality/gadm41_IRL_shp", layer = "gadm41_IRL_1", quiet = TRUE)

df_map <- map %>%
  mutate(
    NAME_1 = as.character(NAME_1),
    # robust to both real NA and the literal "NA" string
    NAME_1 = ifelse(is.na(NAME_1) | NAME_1 == "NA", "Cork", NAME_1)
  ) %>%
  select(county = NAME_1)  # keep county name + geometry

# ------------------ County-level saved lives map -------------------
df_counters <- t(posteriors[["y_counter"]]) %>%
  as.data.frame() %>%
  group_by(county = df_jags$county) %>%
  summarise(across(V1:V6500, ~ sum(.)), .groups = "drop")

df_factuals <- df_jags %>%
  group_by(county) %>%
  summarise(notices = sum(notices), .groups = "drop")

df_count_vs_factual <- left_join(df_counters, df_factuals, by = "county") %>%
  mutate(across(V1:V6500, ~ . - notices))

# row-wise quantiles of posterior draws
df_quantiles <- as.data.frame(
  t(apply(dplyr::select(df_count_vs_factual, V1:V6500), 1,
          function(x) stats::quantile(x, prob = c(0.025, 0.5, 0.975))))
)
# keep county so we can join safely
df_quantiles$county <- df_count_vs_factual$county

 # Join quantiles onto the map (safer than cbind-by-order)
df_map_saved_lives <- left_join(df_map, df_quantiles, by = "county") %>%
  st_as_sf()

# Shared color limits across panels
vals <- st_drop_geometry(df_map_saved_lives) %>% dplyr::select(`2.5%`, `50%`, `97.5%`)
min_saved_lives <- min(unlist(vals))
max_saved_lives <- max(unlist(vals))

p_low <- df_map_saved_lives %>%
  rename(`Number of\nsaved lives` = `2.5%`) %>%
  ggplot() +
  geom_sf(aes(fill = `Number of\nsaved lives`), lwd = 0.1) +
  scale_fill_gradient2(
    midpoint = 0,
    limits = c(min_saved_lives, max_saved_lives),
    low = "red", mid = "white", high = "green4"
  ) +
  ggtitle("Percentile: 2.5%") +
  theme_bw()

p_mid <- df_map_saved_lives %>%
  rename(`Number of\nsaved lives` = `50%`) %>%
  ggplot() +
  geom_sf(aes(fill = `Number of\nsaved lives`), lwd = 0.1) +
  scale_fill_gradient2(
    midpoint = 0,
    limits = c(min_saved_lives, max_saved_lives),
    low = "red", mid = "white", high = "green4"
  ) +
  ggtitle("Percentile: 50%") +
  theme_bw()

p_high <- df_map_saved_lives %>%
  rename(`Number of\nsaved lives` = `97.5%`) %>%
  ggplot() +
  geom_sf(aes(fill = `Number of\nsaved lives`), lwd = 0.1) +
  scale_fill_gradient2(
    midpoint = 0,
    limits = c(min_saved_lives, max_saved_lives),
    low = "red", mid = "white", high = "green4"
  ) +
  ggtitle("Percentile: 97.5%") +
  theme_bw()

ggpubr::ggarrange(p_low, p_mid, p_high, ncol = 3, nrow = 1)
 