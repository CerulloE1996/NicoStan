
#pragma once


  
#include <Eigen/Dense>
 

 
 
 
 
 
using namespace Eigen;
 




#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE






ALWAYS_INLINE bool is_NaN_or_Inf_Eigen(const Eigen::Matrix<double, -1, -1> &mat) {
    
    if (!((mat.array() == mat.array()).all())) {
      return true;
    }   
    
    if ((mat.array().isInf()).any()) {
      return true;
    }   
    
    return false; // if no NaN or Inf values  
  
}  




 





























