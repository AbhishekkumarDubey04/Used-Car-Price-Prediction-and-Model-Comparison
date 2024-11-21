
library(shiny)
library(dplyr)
library(caret)
library(e1071)
library(rpart)
library(class)
library(ggplot2)

# Define UI
ui <- fluidPage(
  titlePanel("Used Car Price Prediction Model Comparison"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Choose CSV File", accept = ".csv"),
      checkboxGroupInput("models", "Choose Models", 
                         choices = c("Linear Regression", "Decision Tree", "KNN", "SVM"),
                         selected = c("Linear Regression", "Decision Tree", "KNN", "SVM")),
      actionButton("run", "Run Models")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Summary", tableOutput("results")),
        tabPanel("Model Comparison", plotOutput("maePlot")),
        tabPanel("Predicted vs Actual", plotOutput("predVsActual"))
      )
    )
  )
)

# Define server
server <- function(input, output) {
  # Reactive value to store results
  results <- reactiveVal(data.frame(Model = character(), MAE = double(), stringsAsFactors = FALSE))
  
  observeEvent(input$run, {
    # Load data
    req(input$file)
    used_cars_data <- read.csv(input$file$datapath)
    
    # Data preprocessing
    data <- used_cars_data %>% 
      dplyr::select(-S.No., -New_Price, -Name) %>% 
      filter(!is.na(Price)) %>% 
      mutate(
        Mileage = as.numeric(sub(" km/kg| kmpl", "", Mileage)), 
        Engine = as.numeric(sub(" CC", "", Engine)), 
        Power = as.numeric(sub(" bhp", "", Power))
      ) %>% 
      na.omit() %>% 
      mutate(across(c(Location, Fuel_Type, Transmission, Owner_Type), as.factor))
    
    # Split data
    set.seed(123)
    trainIndex <- createDataPartition(data$Price, p = 0.8, list = FALSE)
    train <- data[trainIndex, ]
    test <- data[-trainIndex, ]
    
    # Initialize results dataframe
    res <- data.frame(Model = character(), MAE = double(), stringsAsFactors = FALSE)
    pred_vs_actual <- data.frame(Actual = numeric(), Predicted = numeric(), Model = character())
    
    # Run models based on selection
    if ("Linear Regression" %in% input$models) {
      lm_model <- lm(Price ~ ., data = train)
      lm_preds <- predict(lm_model, test)
      res <- rbind(res, data.frame(Model = "Linear Regression", MAE = mean(abs(test$Price - lm_preds))))
      pred_vs_actual <- rbind(pred_vs_actual, data.frame(Actual = test$Price, Predicted = lm_preds, Model = "Linear Regression"))
    }
    
    if ("Decision Tree" %in% input$models) {
      tree_model <- rpart(Price ~ ., data = train)
      tree_preds <- predict(tree_model, test)
      res <- rbind(res, data.frame(Model = "Decision Tree", MAE = mean(abs(test$Price - tree_preds))))
      pred_vs_actual <- rbind(pred_vs_actual, data.frame(Actual = test$Price, Predicted = tree_preds, Model = "Decision Tree"))
    }
    
    if ("KNN" %in% input$models) {
      numeric_train_x <- scale(select_if(train, is.numeric) %>% select(-Price))
      numeric_test_x <- scale(select_if(test, is.numeric) %>% select(-Price))
      knn_preds <- knn(train = numeric_train_x, test = numeric_test_x, cl = train$Price, k = 5)
      knn_preds <- as.numeric(as.character(knn_preds))
      res <- rbind(res, data.frame(Model = "KNN", MAE = mean(abs(test$Price - knn_preds))))
      pred_vs_actual <- rbind(pred_vs_actual, data.frame(Actual = test$Price, Predicted = knn_preds, Model = "KNN"))
    }
    
    if ("SVM" %in% input$models) {
      svm_model <- svm(Price ~ ., data = train)
      svm_preds <- predict(svm_model, test)
      res <- rbind(res, data.frame(Model = "SVM", MAE = mean(abs(test$Price - svm_preds))))
      pred_vs_actual <- rbind(pred_vs_actual, data.frame(Actual = test$Price, Predicted = svm_preds, Model = "SVM"))
    }
    
    # Update results
    results(res)
    
    # Reactive values for plotting
    output$maePlot <- renderPlot({
      ggplot(results(), aes(x = Model, y = MAE, fill = Model)) +
        geom_bar(stat = "identity") +
        labs(title = "Model Comparison (Mean Absolute Error)", y = "Mean Absolute Error", x = "Model") +
        theme_minimal() +
        theme(legend.position = "none")
    })
    
    output$predVsActual <- renderPlot({
      ggplot(pred_vs_actual, aes(x = Actual, y = Predicted, color = Model)) +
        geom_point(alpha = 0.6) +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
        labs(title = "Predicted vs Actual", x = "Actual Price", y = "Predicted Price") +
        facet_wrap(~Model) +
        theme_minimal()
    })
  })
  
  # Output results
  output$results <- renderTable({
    results()
  })
}

# Run the app
shinyApp(ui = ui, server = server)

