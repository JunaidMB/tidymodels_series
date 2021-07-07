library(AmesHousing)
library(janitor)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(finetune)
library(butcher)
library(knitr)
library(remotes)
library(devtools)
#remotes::install_github("curso-r/treesnip")
library(treesnip)
#devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
library(catboost)


# Load Data ----
set.seed(123)
ames_data <- make_ames() %>% 
  janitor::clean_names()

## Partition the dataframe to keep some data to act as new data
ames_future <- tail(ames_data, 100)
ames_current <- head(ames_data, 2830)

# Split Data ----
ames_split <- initial_split(ames_current, prop = 0.8, strata = sale_price)

ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# Create Recipe ----
ames_recipe <- recipe(sale_price ~ ., data = ames_train) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>% 
  step_nzv(all_nominal_predictors())

# Define Model -----
## Add catboost
#treesnip::add_boost_tree_catboost()

catboost_spec <- boost_tree(
  trees = 1000,
  min_n = tune(),
  learn_rate = tune(),
  tree_depth = tune()) %>% 
  set_engine("catboost", loss_function = "RMSE", nthread = 6) %>% 
  set_mode("regression")

## Constrain the learn rate to be between 0.001 and 0.005
catboost_param <- catboost_spec %>% 
  parameters() %>% 
  update(learn_rate = threshold(c(0.001, 0.005)))

# Define Workflow to connect Recipe and Model ----
catboost_workflow <- workflow() %>% 
  add_recipe(ames_recipe) %>% 
  add_model(catboost_spec)

# Train and Tune Model ----
## Define a random grid for hyperparameters to vary over
set.seed(123)

catboost_grid <- grid_max_entropy(
  catboost_param,
  size = 10
)

## Make Cross Validation Folds
set.seed(123)
ames_folds <- vfold_cv(data = ames_train, v = 5)

## Tune Model using Parallel Processing
cores <- parallel::detectCores(logical = FALSE)
cl <- makeForkCluster(cores - 1)  
doParallel::registerDoParallel(cl) # Register Backend

set.seed(123)
catboost_tuned <- catboost_workflow %>% 
  tune_grid(resamples = ames_folds, 
            grid = catboost_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(rmse, rsq, mae))


## View performance metrics across all hyperparameter permutations
catboost_tuned %>% 
  collect_metrics()

# Plot for tuning parameter performance ----
## RMSE
catboost_tuned %>% 
  show_best(metric = "rmse", n = 10) %>% 
  tidyr::pivot_longer(min_n:learn_rate, names_to="variable",values_to="value" ) %>% 
  ggplot(mapping = aes(value, mean)) +
  geom_line(alpha=1/2)+
  geom_point()+
  facet_wrap(~variable, scales = "free")+
  ggtitle("Best parameters for RMSE")

## MAE
catboost_tuned %>% 
  show_best(metric = "mae", n = 10) %>% 
  tidyr::pivot_longer(min_n:learn_rate, names_to="variable", values_to="value" ) %>% 
  ggplot(mapping = aes(value,mean)) +
  geom_line(alpha=1/2)+
  geom_point()+
  facet_wrap(~variable, scales = "free")+
  ggtitle("Best parameters for MAE")

# Finalise Model ----
## Update workflow with the best model found
catboost_best_model <- catboost_tuned %>% 
  select_best(metric = "rmse")

final_catboost_workflow <- catboost_workflow %>% 
  finalize_workflow(catboost_best_model)

## Fit the final model to all the training data
final_catboost_model <- final_catboost_workflow %>% 
  fit(data = ames_train)

# Fit model to test data ----
catboost_fit_final <- final_catboost_model %>% 
  last_fit(ames_split)

# Model Evaluation ----
## Metrics on test set
catboost_fit_final %>% 
  collect_metrics()

## Predictions on test set
catboost_fit_final %>% 
  collect_predictions() %>% 
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(ames_test)

## Residuals on training set
final_catboost_model %>% 
  predict(new_data = ames_train) %>% 
  bind_cols(ames_train) %>% 
  mutate(residuals = sale_price - .pred) %>% 
  ggplot(mapping = aes(x = .pred, y = residuals)) + 
  geom_point()

## Residuals on test set
catboost_fit_final %>% 
  collect_predictions() %>% 
  mutate(residuals = sale_price - .pred) %>% 
  ggplot(mapping = aes(x = .pred, y = residuals)) + 
  geom_point()

# Yardstick prediction metrics ----
## Training set
final_catboost_model %>% 
  predict(new_data = ames_train) %>% 
  bind_cols(ames_train) %>% 
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ",")) %>%
  knitr::kable()

## Test set
catboost_fit_final %>% 
  collect_predictions() %>% 
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ",")) %>%
  knitr::kable()

# Save Workflow ----
## Extract final fitted workflow
catboost_wf_model <- catboost_fit_final$.workflow[[1]]

## Butcher Workflow to reduce extraneous elements in workflow
catboost_wf_model_reduced <- butcher(catboost_wf_model)

## Compute difference between butchered and non butchered workflow objects
lobstr::obj_size(catboost_wf_model)
lobstr::obj_size(catboost_wf_model_reduced)

## Save Model as RDS object
saveRDS(catboost_wf_model_reduced, file = "catboost_saved_model.rds")

# Load Workflow and predict on unseen data ----
catboost_loaded_wf <- readRDS(file = "catboost_saved_model.rds")

## Predict workflow on ames_future data
catboost_loaded_wf %>% 
  augment(ames_future) 

# Stop Cluster ----
parallel::stopCluster(cl)
