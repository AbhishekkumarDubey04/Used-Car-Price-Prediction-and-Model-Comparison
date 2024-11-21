library(dplyr)
library(caret)
library(e1071)      # For SVM and Naive Bayes
library(rpart)      # For Decision Tree
library(class)      # For KNN
library(ggplot2)    # For plotting

# Load the data
used_cars_data <- read.csv(file.choose())
View(used_cars_data)

# Data preprocessing 
data <- used_cars_data %>% 
  dplyr::select(-S.No., -New_Price, -Name) %>% 
  filter(!is.na(Price)) %>% 
  mutate( Mileage = as.numeric(sub(" km/kg| kmpl", "", Mileage)), 
          Engine = as.numeric(sub(" CC", "", Engine)), 
          Power = as.numeric(sub(" bhp", "", Power)) ) %>% na.omit() %>% 
  mutate(across(c(Location, Fuel_Type, Transmission, Owner_Type), as.factor))

#split data
set.seed(123)
trainIndex <- createDataPartition(data$Price, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Initialize results data frame
results <- data.frame(Model = character(), MAE = double(), stringsAsFactors = FALSE)

### Linear Regression
lm_model <- lm(Price ~ ., data = train)
lm_preds <- predict(lm_model, test)
results <- rbind(results, data.frame(Model = "Linear Regression", MAE = mean(abs(test$Price - lm_preds))))

# Plot for Linear Regression
ggplot(data.frame(Actual = test$Price, Predicted = lm_preds), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Linear Regression: Predicted vs Actual", x = "Actual Price", y = "Predicted Price") +
  theme_minimal()

### Decision Tree
tree_model <- rpart(Price ~ ., data = train)
tree_preds <- predict(tree_model, test)
results <- rbind(results, data.frame(Model = "Decision Tree", MAE = mean(abs(test$Price - tree_preds))))

# Plot for Decision Tree
ggplot(data.frame(Actual = test$Price, Predicted = tree_preds), aes(x = Actual, y = Predicted)) +
  geom_point(color = "green") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Decision Tree: Predicted vs Actual", x = "Actual Price", y = "Predicted Price") +
  theme_minimal()

### K-Nearest Neighbors - only numeric columns scaled
numeric_train_x <- scale(select_if(train, is.numeric) %>% select(-Price))
numeric_test_x <- scale(select_if(test, is.numeric) %>% select(-Price))

knn_preds <- knn(train = numeric_train_x, test = numeric_test_x, cl = train$Price, k = 5)
knn_preds <- as.numeric(as.character(knn_preds))
results <- rbind(results, data.frame(Model = "KNN", MAE = mean(abs(test$Price - knn_preds))))

# Plot for KNN
ggplot(data.frame(Actual = test$Price, Predicted = knn_preds), aes(x = Actual, y = Predicted)) +
  geom_point(color = "purple") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "KNN: Predicted vs Actual", x = "Actual Price", y = "Predicted Price") +
  theme_minimal()

### Support Vector Machine
svm_model <- svm(Price ~ ., data = train)
svm_preds <- predict(svm_model, test)
results <- rbind(results, data.frame(Model = "SVM", MAE = mean(abs(test$Price - svm_preds))))

# Plot for SVM
ggplot(data.frame(Actual = test$Price, Predicted = svm_preds), aes(x = Actual, y = Predicted)) +
  geom_point(color = "orange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "SVM: Predicted vs Actual", x = "Actual Price", y = "Predicted Price") +
  theme_minimal()

# Print results
print(results)

# Final MAE Comparison Plot
ggplot(results, aes(x = Model, y = MAE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison (Mean Absolute Error)", y = "Mean Absolute Error", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")