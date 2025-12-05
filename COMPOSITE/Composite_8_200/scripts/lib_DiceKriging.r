# DiceKriging

switch(ACTION,
  "init" = {
    # INITIALIZE DiceKriging
    library(DiceKriging)
    # Convert data to appropriate formats
    train_x <<- as.matrix(train_x)
    train_y <<- as.numeric(train_y)
    test_x <<- as.matrix(test_x)
    # Set dimensionality for 8D input
    d <<- ncol(train_x)  # d = 8
  },
  "train" = {
    # TRAINING
    model <<- km(
      formula = ~1, # Constant trend
      design = train_x,
      response = train_y,
      covtype = "gauss", # Gaussian/SE covariance
      nugget = 1e-8,  # small fixed nugget
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
