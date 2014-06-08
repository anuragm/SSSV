//
//  runSSSV.cpp
//  SSSV
//
//  Created by Anurag Mishra on 5/21/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.
//

#include <fstream>
#include <vector>
#include <armadillo>
#include "runSSSV.hpp"

using namespace arma;

vec runSSSV(vec h, mat J, int numOfSweeps, double temperature, const mat& schedule)
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
                randomAngle = drand48()*datum::pi;  //generate a random angle between 0 and pi
                energyDiff =   magB * ( cos(randomAngle) - cos(theta(iiQubits)) )*(h(iiQubits) + as_scalar( J.row(iiQubits)*cos(theta) ) )
                - magA * ( sin(randomAngle) - sin(theta(iiQubits)) );
                probToFlip = exp(-energyDiff/temperature);
                
                if(drand48() < probToFlip)
                    theta(iiQubits) = randomAngle;
            }
        }
    }
    
    return theta;
}

void getSigHam(double scale, double variance, arma::vec* h, arma::mat* J)
{
    int numOfQubits = h->n_elem;
    
    //First we fill h with random values.
    for (int i=0; i<numOfQubits; i++)
        (*h)(i) = nrand48(variance);
    
    h->subvec(0,numOfQubits/2-1) += arma::vec(4).fill(scale); //core qubits, add +1
    h->subvec(numOfQubits/2,numOfQubits-1) += arma::vec(4).fill(-scale); //auxillary qubits, add -1
    
    //then we fill the J matrix.
    //core-core links
    for (int i=0; i<numOfQubits/2-1; i++)
        (*J)(i,i+1) = scale + nrand48(variance);
    (*J)(0,numOfQubits/2-1) = scale + nrand48(variance);
    
    //core-ancilla links
    for(int i=0;i<numOfQubits/2;i++)
        (*J)(i,i+numOfQubits/2) = scale + nrand48(variance);
    
    //make J symettric.
    *J = *J + J->t();
    
}

double nrand48(double variance) //generates normal number via Box-muller method
{
    static double rand1, rand2 ; //two random number, static so that they are remembered across function calls.
    static bool usedBoth = true; //as we get two normal random number each time, we can return the spare one on second call
    
    if(!usedBoth)
    {
        usedBoth = true;
        return sqrt(variance*rand1)*sin(rand2); //return the second normal number that we hadn't used
    }
    
    //If control reaches here, it means we have used both random numbers, and we need to generate fresh ones.
    usedBoth = false;
    rand1 = drand48();
    if (rand1<1e-100) {
        rand1=1e-100;
    }
    
    rand1 = -2*log(rand1);
    rand2 = 2*datum::pi*drand48();
    
    return sqrt(variance*rand1)*cos(rand2); //returns one of the random number, the other is still ramianing.
}

void readHamiltonian(arma::vec* h,arma::mat* J, const std::string& fileName) //Reads the Hamiltonian from a file.
{
    //The format is assumed to be the DW2 input format, that is,
    //number of qubits and total number of lines in first line, and then the rest of couplings.
    
    std::ifstream hamFile;
    
    if (fileName.empty())
        hamFile.open("hamiltonian.config");
    else
        hamFile.open(fileName.c_str());
    
    if(!hamFile.is_open()) //If file is not opened, return silently.
    {
        std::cerr<<"cannot read from Hamiltonian file. Check if the file exists and is readable \n";
        std::logic_error("Cannot read Hamiltonian file");
        return;
    }
    
    //Read the first line into total number of qubits and total number of lines.
    int numOfQubit; int totalLines;
    hamFile>>numOfQubit>>totalLines;
    
    h->zeros(numOfQubit); J->zeros(numOfQubit,numOfQubit); //Resize h and J to given number of qubits.
    //Now, for each line, read the file into h and J's.
    for (int ii=0; ii<totalLines; ii++)
    {
        int location1, location2;
        hamFile>>location1>>location2;
        if (location1==location2)
            hamFile>>(*h)(location1);
        else
        {
            //make sure J is initialized as upper triangle matrix
            int rowLocation = (location1<location2)?location1:location2;
            int colLocation = location1+location2-rowLocation;
            hamFile>>(*J)(rowLocation,colLocation);
        }
    }
    
    //Done!. Care must be taken to make J symettric.
    *J = *J + J->t();
}


void readParameters(int* numOfSSSVRuns, int* numOfSweeps, double* temperature, double* noise, const std::string& fileName)
{
    std::ifstream paramFile;
    if (fileName.empty())
        paramFile.open("SSSV.config");
    else
        paramFile.open(fileName);
    
    if (!paramFile.is_open()) {
        std::cerr<<"Cannot find SSSV parameters file. Check if the file exists. ";
        throw std::logic_error("No SSSV Config file");
        return;
    }
    
    paramFile>>*numOfSSSVRuns>>*numOfSweeps>>*temperature>>*noise;
}

arma::vec getScalings(const std::string& fileName)
{
    std::ifstream scalingFile;
    if (fileName.empty())
        scalingFile.open("scaling.config");
    else
        scalingFile.open(fileName);
    
    if(!scalingFile.is_open()) //If file is not opened, return 1 with warning message.
    {
        std::cerr<<"couldn't read scaling file. Default scaling factor of 1 used.";
        arma::vec scalings("1");
        return scalings;
    }
    
    double readANumber;
    std::vector<double> scaling;
    
    while (true) {
        scalingFile>>readANumber;
        if(scalingFile.eof())
            break;
        scaling.push_back(readANumber);
    }
    
    return scaling; //std::vector is automatically type casted to aram::vec.
}







