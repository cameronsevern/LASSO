#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# list.of.packages <- c("shiny", "DT","tidyverse", "caret","elasticnet","lime","doParallel","xgboost","simputation")
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
# if(length(new.packages)) install.packages(new.packages)

library(shiny); library(DT);library(tidyverse);library(caret);library(glmnet);library(doParallel);library(simputation);library(pROC)
library(shinyWidgets)


# Define UI for application that draws a histogram
ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "style.css"),
    tags$style(HTML("#fitMetricsTab tr, #coefsTab tr, #cleanDataTab tr, #dataTab tr {background-color:#f2e0b960}"))
  ),
  setBackgroundImage(src = "./images/background.jpg", shinydashboard = FALSE),
  navbarPage(title = "LASSO",
             tabPanel(title = "Howdy",
                      h1("Howdy, partner!"),
                      div(style="display: inline-block;vertical-align:top; width: 49%;",
                          h2("What in tarnation?"),
                          p("In statistics and machine learning, LASSO (least absolute shrinkage and selection operator) 
                         is a regression analysis method that performs both variable selection and regularization in order to enhance the 
                         prediction accuracy and interpretability of the resulting statistical model."),
                          h2("Instructions"),
                          h3("1. Upload your data"),
                          p("Head over to the Data Wranglin' section using the navigation bar and select a file. Your data should be in CSV format and have variable names as the first row."),
                          h3("2. Wrangle it"),
                          p("The data need to be formatted in order to be LASSO'd correctly. Drop any variables that you don't want to be considered as predictors, 
                        and be sure to create dummy variables for any categorical variables. It is recommended to center and scale the numerical values and drop any variables with little
                        to no variance. LASSO requires there be no missing data, so you can choose to drop rows with missing data or impute."),
                          h3("3. Yeehaw!"),
                          p("Pick a variable you'd like to predict and then run the LASSO by clicking 'Giddy Up', you'll see information about 
                        how the model was selected and which variables were important. You may chose to do further analysis with the selected 
                        variables, or right-click to save the plots. LASSOMAN uses repeated k-fold cross-validation to avoid overfitting and to estimate the variability of the prediction accuracy")
                          
                      ),
                      div(style="display: inline-block;vertical-align:top; width: 49%;",
                          tags$div(img(src = "./images/cowboy.jpg"))
                      ),
                      
             ),
             tabPanel(title = "Data Wranglin'",
                      sidebarPanel(
                        fileInput(inputId = "dataUpload",
                                  label = "Upload Data",
                                  accept = "csv"),
                        selectInput("impute", "Imputation Method", choices = c("None (Drop Incomplete Cases)" = "none",
                                                                               "Median" = "medianImpute"),
                                    selected = "medianImpute"),
                        selectInput("dropVars", "Drop Variables", choices = NULL, multiple = T),
                        selectInput("makeFactor", "Make Factor", choices = NULL, multiple = T),
                        selectInput("dummies", "Dummy Variables", choices = c("None" = "none",
                                                                              "Full Rank" = "fullRank",
                                                                              "All Levels" = "ltFullRank"),
                                    selected = "ltFullRank"),
                        
                        checkboxGroupInput(
                          inputId = "preProcess",
                          label = "Pre-Processing Options",
                          choices = c("Center" = "center",
                                      "Scale" = "scale",
                                      "Remove Zero Variance Variables" = "zv",
                                      "Remove Near-Zero Variance Variables" = "nzv"),
                          selected = c("center","scale","zv","nzv"),
                          inline = FALSE
                        )
                        
                      ),
                      mainPanel(
                        tabsetPanel(
                          tabPanel("Clean Data", DT::dataTableOutput("cleanDataTab")),
                          tabPanel("Raw Data", DT::dataTableOutput("dataTab"))
                        )
                      )
                      
             ),
             tabPanel(title = "The LASSO",
                      sidebarPanel(
                        selectInput("outcomeVar", "Outcome", choices = NULL),
                        selectInput("outcomeType","Outcome Type", choices = c("Binary" = "binomial","Continuous" = "gaussian")),
                        selectInput("dropVarsPost", "Drop Variables", choices = NULL, multiple = T),
                        numericInput(inputId = "kfold",
                                     label = "k",
                                     value = 5,
                                     min = 3,
                                     step = 1
                        ),
                        numericInput(inputId = "seed",
                                     label = "Set Random Seed",
                                     value = 66,
                                     min = 1,
                                     step = 1
                        ),
                        actionButton(inputId = "lassoRun", label = "Giddy Up!")
                        
                      ),
                      mainPanel(
                        div(style="display: inline-block;vertical-align:top; width: 49%;",plotOutput("performancePlot", height = "25vh")),
                        div(style="display: inline-block;vertical-align:top; width: 49%;",plotOutput("coefPlot", height = "25vh")),
                        
                        div(style="display: inline-block;vertical-align:top; width: 100%;",plotOutput("fitMetricsPlot", height = "25vh")),
                        
                        tabsetPanel(
                          tabPanel(title = "Best Model Parameters",
                                   DT::dataTableOutput("fitMetricsTab")
                                   ),
                          tabPanel(title = "All Coefficients",
                                   DT::dataTableOutput("coefsTab")
                            
                          )
                        )
                        
                        
                      )
                      
             )
             # tabPanel(title = "Tying Things Up",
             #          sidebarPanel(),
             #          mainPanel()
             #          
             # )
  )
  
  
  
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  ## Increasing max input file size
  options(shiny.maxRequestSize=30*1024^2)
  
  dataUpload <- reactive({
    req(input$dataUpload)
    vroom::vroom(input$dataUpload$datapath,
                 .name_repair = 'minimal')
  })
  
  cleanData <- reactive({
    validate(
      need(nrow(dataUpload()) > 0, "Invalid dataset")
    )
    if (length(input$dropVars) > 0){
      df <- dataUpload() %>% select(-input$dropVars)
    }
    else {
      df <- dataUpload()
    }
    
    if (length(input$makeFactor) > 0){
      for (fac in input$makeFactor){
        df[fac] <- as.factor(df[[fac]])
      }
    }
    
    
    
    methods <- NULL
    
    if (input$impute == "medianImpute"){
      df <- impute_median(df, .~1)
      # } else if (input$impute == "knnImpute"){
      #   methods <- c(methods, "knnImpute")
    } else
      validate(
        need(nrow(df[complete.cases(df),]) > 0, "All rows removed. Consider removing variables or imputing.")
      )
    
    df <- df[complete.cases(df),]
    
    if (length(input$preProcess) > 0){
      methods <- c(methods, input$preProcess)
      
    }
    
    if (length(methods) > 0) {
      preProcValues  <- preProcess(df, method = methods)
      # validate(
      #   need(preProcValues$dim[1] > 0, "All rows removed. Please adjust preprocessing actions.")
      # )
      df <- predict(preProcValues, df)
    }
    
    if (input$dummies == "fullRank"){
      df <- model.matrix( ~ ., data = df)
    } else if (input$dummies == "ltFullRank"){
      dummies <- dummyVars( ~ ., data = df)
      df <- predict(dummies, newdata = df)
    }
    
    
    
    df <- as.data.frame(df)
    #saveRDS(df, file = "clean_data.RData")
    df
    
    
  })
  
  analysisData <- reactive({
    if (length(input$dropVarsPost) > 0){
      df <- cleanData() %>% select(-input$dropVarsPost)
    }
    else {
      df <- cleanData()
    }
  })
  
  fit <- eventReactive(input$lassoRun,{
    df <- analysisData()
    
    cl <- makePSOCKcluster(parallel:::detectCores()-1)
    registerDoParallel(cl)
    
    # if (length(unique(df[[input$outcomeVar]])) == 2){
    #   df[[input$outcomeVar]] <- as.factor(df[[input$outcomeVar]])
    # }
    
    
    y <- df[[input$outcomeVar]]

    X <- makeX(df[names(df)[!names(df) %in% input$outcomeVar]])

    
    results <- NULL
    i <- 1
    
    withProgress(message = "Fitting Models", expr = 
      {
        for (alpha in seq(0.1,1,0.1)){
          set.seed(1)
          model <- cv.glmnet(y = y, x = X, family = input$outcomeType, relax = T, alpha = alpha, parallel = T, nfolds = as.numeric(input$kfold))
          assign(paste("model",i, sep = "_"), model)
          cv_search <- as.data.frame(model$relaxed$statlist) %>% select(g.0.lambda, ends_with("cvm")) %>% pivot_longer(cols = ends_with("cvm"),
                                                                                                                       names_pattern = "g.(.*).cvm",
                                                                                                                       names_to = "gamma",
                                                                                                                       values_to = "cvm")
          names(cv_search) <- c("lambda","gamma", "deviance")
          
          cv_search$alpha <- alpha
          cv_search$model <- i
          
          results <- rbind(results, cv_search)
          setProgress(alpha)
          i <- i + 1
        }
      }
    )
    
    results$ismin <- results$deviance == min(results$deviance)
    results_min <- results[results$ismin,]
    
    results_metrics <- results_min %>% select(alpha, gamma, lambda, deviance)
    
    results_plot <- ggplot(data = results, aes(y  = deviance, x = lambda, color = as.factor(alpha))) +
      geom_line() +
      geom_point() +
      theme_bw() +
      facet_grid(~gamma, labeller = label_both) +
      geom_point(inherit.aes = F, data = results_min, aes(y  = deviance, x = lambda, color = as.factor(alpha)), color = "black", shape = 18, size = 3) +
      labs(y = "deviance", x = "lambda", color = "alpha") +
      ggtitle("Hyperparameter Tuning") +
      theme(
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        legend.background = element_rect(fill = "transparent",colour = NA),
        legend.box.background = element_rect(fill = "transparent",colour = NA),
        strip.background = element_rect(fill = "transparent",colour = NA)
      )
    
    best_model <- get(paste("model",results_min$model[1],sep = "_"))
    
    results_metrics$r_squared <- best_model$glmnet.fit$dev.ratio[model$index[1]]
    
    coef_mat <- as.matrix(coef.glmnet(best_model, s = "lambda.min"))
    
    result_coefs <- data.frame(var = rownames(coef_mat)[coef_mat != 0], coefficient = coef_mat[coef_mat != 0])
    
    result_coefs <- result_coefs[-1,]
    
    result_coefs <- result_coefs %>% arrange(desc(abs(coefficient))) %>% na.omit()
    
    results_coefs <- result_coefs[result_coefs$var != "NA",]
    
    results_coefs <- result_coefs[!is.na(result_coefs$var),]
    
    result_coefs$var <- factor(result_coefs$var, levels = rev(result_coefs$var))
    
    coefs_plot <- ggplot(data = result_coefs[1:10,], aes(x = coefficient, y = var)) +
      geom_bar(stat = "identity") +
      theme_bw() +
      ggtitle("Variable Coefficients") +
      theme(
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA)
      )
    
    
    
    if (input$outcomeType == "binomial"){
      preds <- predict(best_model,newx = X, type = "response")
      rocobj <- roc(y, preds)
      performance_plot <- ggroc(rocobj) + 
        theme_bw() +
        ggtitle("ROC-AUC") +
        theme(
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA)
        )
    }
    else if(input$outcomeType == "gaussian"){
      preds <- predict(best_model,newx = X)
      performance_plot <- ggplot(data = data.frame(preds = preds, y = y)) +
        geom_point(aes(y = preds, x = y)) +
        labs(x = "Actual", y = "Predicted") + 
        geom_abline(slope = 1) +
        ggtitle("Residuals") +
        theme_bw() +
        theme(
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA)
        )
    }
    
    stopCluster(cl)
    
    list(results_plot = results_plot, coefs_plot = coefs_plot, results_metrics = results_metrics, performance_plot = performance_plot, result_coefs = result_coefs)
    
  })
  
  
  output$dataTab <- DT::renderDataTable({
    validate(
      need(nrow(dataUpload()) > 0, "No valid data found. Please check uploaded spreadsheet for errors.")
    )
    DT::datatable(dataUpload(),options = list(
      pageLength = 100,
      scrollY = "70vh",scrollX = TRUE))
  })
  
  output$cleanDataTab <- DT::renderDataTable({
    validate(
      need(nrow(cleanData()) > 0, "No complete rows in data set. Please impute or remove variables with missing values.")
    )
    DT::datatable(cleanData(),options = list(
      pageLength = 100, 
      scrollY = "70vh",scrollX = TRUE))
  })
  
  output$fitMetricsTab <- renderDataTable({
    
    DT::datatable(fit()$results_metrics, caption = "Best Model Parameters")
    
  })
  
  output$coefsTab <- renderDataTable({
    
    DT::datatable(fit()$result_coefs, caption = "Model Coefficients")
    
  })
  
  
  output$fitMetricsPlot <- renderPlot({
    req(fit())
    fit()$results_plot
  }, bg="transparent")
  
  
  output$coefPlot <- renderPlot({
    req(fit())
    fit()$coefs_plot
  }, bg="transparent")
  
  output$performancePlot <- renderPlot({
    req(fit())
    fit()$performance_plot
  }, bg="transparent")
  
  
  
  observeEvent(dataUpload(),
               updateSelectInput(session, "dropVars",
                                 choices = names(dataUpload()) ))
  
  
  observeEvent(dataUpload(),
               updateSelectInput(session, "makeFactor",
                                 choices = names(dataUpload()) ))
  
  observeEvent(req(cleanData()),{
    updateSelectInput(session, "outcomeVar", choices = names(cleanData()))
  })
  
  observeEvent(cleanData(),
               updateSelectInput(session, "dropVarsPost",
                                 choices = names(cleanData()) ))
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)
