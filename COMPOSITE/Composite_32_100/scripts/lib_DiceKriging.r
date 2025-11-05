# DiceKriging

switch(ACTION,
  "init" = {
    # INITIALIZE DiceKriging
    library(DiceKriging)
    # Convert data to appropriate formats
    train_x <<- as.matrix(train_x)
    train_y <<- as.numeric(train_y)
    test_x <<- as.matrix(test_x)
  },
  "train" = {
    # TRAINING
    model <<- km(
      formula = ~1, # Zero trend (~1 for constant trend)
      design = train_x,
      response = train_y,
      covtype = "gauss", # Squared exponential covariance
      nugget = 1e-10,
      optim.method = "BFGS",
      control = list(trace = FALSE) # Suppress optimization output
    )
  },
  "test" = {
    # PREDICTION
    pred <- predict(
      model,
      newdata = test_x,
      type = "UK", # Universal Kriging
      se.compute = TRUE
    )
    # Extract predictions and variances
    pred_y <<- pred$mean
    pred_var <<- pred$sd^2
  }
)
