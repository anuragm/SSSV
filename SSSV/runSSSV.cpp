//
//  runSSSV.cpp
//  SSSV
//
//  Created by Anurag Mishra on 5/21/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.
//

#include <armadillo>
#include "runSSSV.hpp"

using namespace arma;

vec runSSSV(vec h, mat J, int numOfSweeps, double temperature, mat schedule)
{
    //Assumptions: The J matrix is symettric. Or the code would output wrong results.
    
    int numOfQubits = h.n_elem;
    vec theta(numOfQubits);
    theta.fill(datum::pi/2);        //Initialize angle vector to pi/2.
    
    int iiTime, iiSweep, iiQubits;  //Counters for time, sweep and qubit.
    //Number of time steps is 1000 (just to make the thing easier for now).
    double magA, magB,randomAngle, energyDiff, probToFlip;
    
    for(iiTime=0;iiTime<schedule.n_rows;iiTime++)
    {
        magA = schedule(iiTime,1); magB = schedule(iiTime,2);
        for(iiSweep=0;iiSweep<numOfSweeps;iiSweep++)
        {
            for(iiQubits=0;iiQubits<numOfQubits;iiQubits++)
            {
                randomAngle = double(rand())/RAND_MAX*datum::pi;  //generate a random angle between 0 and pi
                energyDiff =   magB * ( cos(randomAngle) - cos(theta(iiQubits)) )*(h(iiQubits) + as_scalar( J.row(iiQubits)*cos(theta) ) )
                - magA * ( sin(randomAngle) - sin(theta(iiQubits)) );
                probToFlip = exp(-energyDiff/temperature);
                
                if(double(rand())/RAND_MAX < probToFlip)
                    theta(iiQubits) = randomAngle;
            }
        }
    }
    
    return theta;
}
