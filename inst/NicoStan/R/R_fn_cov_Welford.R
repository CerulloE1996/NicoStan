



#' #' update_cov_Welford
#' @keywords internal
#' @export
update_cov_Welford <- function( 
                                delta_old, # delta between new sample and old mean
                                delta_new, # delta between new sample and new mean
                                ii, 
                                cov_mat) {
  
        try({  
              
              ##  cov_mat <- ((ii-1)/ii) * cov_mat + (1/ii) * (delta_old %*% t(delta_new))
              cov_mat <- ((ii-2)/(ii-1)) * cov_mat + (1/(ii-1)) * (delta_old %*% t(delta_new)) ## for unbiased estimate
              ## cov_mat <- Rcpp_near_PD(cov_mat)
        })
        
        return(cov_mat)
  
}






