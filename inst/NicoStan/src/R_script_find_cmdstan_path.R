

PKG_SRC_DIR <- getwd() # package SRC directory 
PKG_ROOT_DIR <- dirname(PKG_SRC_DIR) # Go up one level to package ROOT directory
PKG_R_DIR <- file.path(PKG_ROOT_DIR, "R") # Go into package R directory

source(file.path(PKG_R_DIR, "R_fn_find_cmdstan_path.R"))

USER_CMDSTAN_DIR <- cmdstanr_path()
cat(USER_CMDSTAN_DIR)
