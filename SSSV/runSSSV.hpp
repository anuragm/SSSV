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

arma::vec runSSSV(arma::vec h, arma::mat J, int numOfSweeps, double temperature, arma::mat schedule);

template<class RNG, class DIST>
void getSigHam(double scale, RNG& rng, DIST& distribution, arma::vec* h, arma::mat* J);

//----------------------------------------------------------------------------------------//


//Template implementations
template<class RNG, class DIST> //a template has been used to handle different types of random number generators and distributions, say mersenne-twister with normal distribution.
void getSigHam(double scale, RNG& rng, DIST& distribution, arma::vec* h, arma::mat* J)
{
    int numOfQubits = h->n_elem;
    
    //First we fill h with random values.
    for (int i=0; i<numOfQubits; i++)
        (*h)(i) = distribution(rng);

    h->subvec(0,numOfQubits/2-1) += arma::vec(4).fill(scale); //core qubits, add +1
    h->subvec(numOfQubits/2,numOfQubits-1) += arma::vec(4).fill(-scale); //auxillary qubits, add -1
    
    //then we fill the J matrix.
    //core-core links
    for (int i=0; i<numOfQubits/2-1; i++)
        (*J)(i,i+1) = scale + distribution(rng);
    (*J)(0,numOfQubits/2-1) = scale + distribution(rng);
    
    //core-ancilla links
    for(int i=0;i<numOfQubits/2;i++)
        (*J)(i,i+numOfQubits/2) = scale + distribution(rng);
}

#endif