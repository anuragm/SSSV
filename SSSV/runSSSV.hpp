//  runSSSV.hpp
//  SSSV
//
//  Created by Anurag Mishra on 5/21/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.
//

#ifndef SSSV_runSSSV_hpp
#define SSSV_runSSSV_hpp

#include <armadillo>

arma::vec runSSSV(arma::vec h, arma::mat J, int numOfSweeps, double temperature, arma::mat schedule);
void sigHam(int numOfQubits, double scale, arma::vec& h, arma::mat& J);

#endif
