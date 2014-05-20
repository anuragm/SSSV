//  main.cpp
//  SSSV
//
//  Created by Anurag Mishra on 5/14/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.

// I want to rewrite SSSV as C++ code so that I can run it in parallel on HPCC without too much stress.
// For now, let's put everything in the main file. Later, we can split them into different functions.

#include <iostream>
#include <armadillo>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace arma;

vec runSSSV(vec h, mat J, int numOfSweeps, double temperature);

int main(int argc, const char * argv[])
{
    srand(time(NULL)); //reinitialize random number generator everytime the function is called?
    //---------------------------------------------------------------//
    //Declare h and J
    mat J(8,8) ; //declares a matrix J of type double, and of size 8x8
    //define all the links for J
    J(0,4) = J(1,5) = J(2,6) = J(3,7) = 1; // core ancilla links
    J(0,1) = J(1,2) = J(2,3) = J(1,3) = 1; // core core links.
    
    vec h(8) ; // a column vector of eight elements.
    h.subvec(0,3).fill(1); //the core qubits have local field +1
    h.subvec(4,7).fill(-1); //the ancilla qubits have local field -1
    
    //---------------------------------------------------------------//
    
    
    int numOfSSSVRuns  = 1000; //Number of times SSSV should be run.
    int numOfSweeps    = 5;
    double temperature = 2.226;
    
    mat allAngles(8,numOfSSSVRuns);
    
    for (int iiRuns=0; iiRuns < numOfSSSVRuns;iiRuns++)
    {
        allAngles.col(iiRuns) = runSSSV(-h,-J,numOfSweeps,temperature);
    }
    
    //save the output in some file.
    allAngles.save("allAngles.txt",raw_ascii);
    
    //Convert to binary vectors and save as file as well. This file is numOfRuns rows, 8 columns. (Easier to read on Mac)
    imat allSpins = trans(conv_to<imat>::from(allAngles > datum::pi/2));
    allSpins.save("allSpins.txt",raw_ascii);
    
    return 0;
}

vec runSSSV(vec h, mat J, int numOfSweeps, double temperature)
{
    //Assumptions: The J matrix send in is in either upper column format, or lower column format
    //the code would fail otherwise.
    
    int numOfQubits = h.n_elem;
    vec theta(numOfQubits);
    theta.fill(datum::pi/2); //Initialize angle vector to pi/2.
    
    //load the DW2 schedule.
    mat dw2schedule;
    dw2schedule.load("dw2schedule.txt",raw_ascii);
    
    
    int iiTime, iiSweep, iiQubits;  //counters for time, sweep and qubit.
                                    //Number of time steps is 1000 (just to make the thing easier for now).
    double magA, magB,randomAngle, energyDiff, probToFlip;
    
    for(iiTime=0;iiTime<dw2schedule.n_rows;iiTime++)
    {
        magA = dw2schedule(iiTime,1); magB = dw2schedule(iiTime,2);
        for(iiSweep=0;iiSweep<numOfSweeps;iiSweep++)
        {
            for(iiQubits=0;iiQubits<numOfQubits;iiQubits++)
            {
                randomAngle = double(rand())/RAND_MAX*datum::pi;  //generate a random angle between 0 and pi
                energyDiff =   magB * ( cos(randomAngle) - cos(theta(iiQubits)) )*(h(iiQubits) + as_scalar( J.row(iiQubits)*cos(theta) ) )
                             - magA * ( sin(randomAngle) - sin(theta(iiQubits)));
                probToFlip = exp(-energyDiff/temperature);
                
                if(double(rand())/RAND_MAX < probToFlip)
                    theta(iiQubits) = randomAngle;
            }
        }
    }
    
    return theta;
}
