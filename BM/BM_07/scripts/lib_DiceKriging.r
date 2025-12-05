# DiceKriging

switch(ACTION,
  "init" = {
    # INITIALIZE DiceKriging
    library(DiceKriging)
    # Convert data to appropriate formats
    train_x_mat <- as.matrix(train_x)
    train_y_vec <- as.numeric(train_y)
    test_x_mat <- as.matrix(test_x)
    
    # Normalize inputs
    mu_x <<- colMeans(train_x_mat)
    sigma_x <<- apply(train_x_mat, 2, sd)
    sigma_x[sigma_x == 0] <<- 1.0
    train_x_norm <<- scale(train_x_mat, center = mu_x, scale = sigma_x)
    
    # Normalize target
    mu_y <<- mean(train_y_vec)
    sigma_y <<- sd(train_y_vec)
    if (sigma_y == 0) sigma_y <<- 1.0
    train_y_norm <<- (train_y_vec - mu_y) / sigma_y
    
    # Set dimensionality for 20D input (Morris)
    d <<- ncol(train_x_mat) 
  },
  "train" = {
    # TRAINING
    model <<- km(
      formula = ~1, # Constant trend
      design = train_x_norm,
      response = train_y_norm,
      covtype = "matern5_2", # Matérn 5/2 covariance
      nugget = 1e-5,  # small nugget (normalized)
      optim.method = "BFGS",
      control = list(trace = FALSE) # Suppress optimization output
    )
  },
  "test" = {
    # PREDICTION
    # Normalize test data
    test_x_norm <- scale(test_x_mat, center = mu_x, scale = sigma_x)
    
    pred <- predict(
      model,
      newdata = test_x_norm,
      type = "UK", # Universal Kriging
      se.compute = TRUE
    )
    # Extract predictions and variances and denormalize
    pred_y <<- pred$mean * sigma_y + mu_y
    pred_var <<- (pred$sd^2) * (sigma_y^2)
  }
)
