//  runSSSV.hpp
//  SSSV
//
//  Created by Anurag Mishra on 5/21/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.
//

#ifndef SSSV_runSSSV_hpp
#define SSSV_runSSSV_hpp

#include <armadillo>

//definitions

arma::vec runSSSV(arma::vec h, arma::mat J, int numOfSweeps, double temperature, const arma::mat& schedule);
void getSigHam(double scale, double variance, arma::vec* h, arma::mat* J);
double nrand48(double variance); 

//----------------------------------------------------------------------------------------//
//template functions and classes



#endif