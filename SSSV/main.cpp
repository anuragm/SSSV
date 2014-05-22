//  main.cpp
//  SSSV
//
//  Created by Anurag Mishra on 5/14/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.

#include <iostream>
#include <armadillo>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "runSSSV.hpp"

using namespace std;
using namespace arma;

#define MASTER 0    // defines the master thread, or the root node.

int main(int argc, char * argv[])
{
    //Common parameters for all threads
    int NumOfSSSVRuns  = 1000;  //Number of times SSSV should be run.
    int numOfSweeps    = 5;
    double temperature = 1.383; //Temperature used by Shin et al
    
    MPI::Init(argc,argv);                          //Initialize openMPI
    int numOfThreads = MPI::COMM_WORLD.Get_size(); //Tells the total number of thread availible.
    int node_id      = MPI::COMM_WORLD.Get_rank(); //Gives the id of the current thread
    
    srand(time(NULL)+node_id); //initialize random number generator differently for each node
    
    //---------------------------------------------------------------//
    int numOfQubits = 8; //identify total number of qubits in the simulation.
    
    //Declare h and J
    mat J(8,8) ; //declares a matrix J of type double, and of size 8x8
    //define all the links for J
    J(0,4) = J(1,5) = J(2,6) = J(3,7) = 1; // core ancilla links
    J(0,1) = J(1,2) = J(2,3) = J(1,3) = 1; // core core links.
    J = J + J.t();                         // make J symettric.
    
    vec h(8) ; // a column vector of eight elements.
    h.subvec(0,3).fill(1); //the core qubits have local field +1
    h.subvec(4,7).fill(-1); //the ancilla qubits have local field -1
    //---------------------------------------------------------------//
    
    //Num of jobs done by current thread.
    int numOfJobs = NumOfSSSVRuns/numOfThreads + ((node_id<NumOfSSSVRuns%numOfThreads)?1:0) ;
    
    //Load the required schedule.
    mat dw2schedule;
    dw2schedule.load("dw2schedule.txt",raw_ascii);
    
    //Auxillary threads broascast their calculations.
    if(node_id!=MASTER)
    {
        cout<<"I am thread "<<node_id<<" and I will do "<<numOfJobs<<" jobs"<<endl;
        //run each job one by one, and send the result to master node.
        for (int iiRuns=0; iiRuns < numOfJobs;iiRuns++)
        {
            vec VecAngles = runSSSV(-h,-J,numOfSweeps,temperature,dw2schedule);
            //convert vector to an double array of size numOfQubits
            double ArrayAngles[numOfQubits];
            memcpy(ArrayAngles, VecAngles.memptr(), numOfQubits*sizeof(double));
            
            //send the resultant array to master node
            MPI::COMM_WORLD.Send(ArrayAngles,numOfQubits,MPI::DOUBLE,MASTER,iiRuns);
        }
    }
    
    //Master node collects the data, and saves them to disk.
    if(node_id==MASTER)
    {
        cout<<"I am thread "<<node_id<<" and I will do "<<numOfJobs<<" jobs"<<endl;
        //Run master nodes jobs.
        mat allAngles(numOfQubits,NumOfSSSVRuns);
        int runCount =0;
        for(int i=0;i<numOfJobs;i++)
        {
            allAngles.col(runCount)=runSSSV(-h, -J, numOfSweeps, temperature,dw2schedule);
            runCount++;
        }
        //Receive jobs from other nodes.
        for(int j=1;j<numOfThreads;j++)
        {
            //calculate number of jobs done by that node.
            int j_NumOfJobs = NumOfSSSVRuns/numOfThreads + ((j<NumOfSSSVRuns%numOfThreads)?1:0);
            //receive all data send from thread j, and convert it into a column of allAngles
            for(int i=0;i<j_NumOfJobs;i++)
            {
                double ArrayAngles[numOfQubits];
                MPI::COMM_WORLD.Recv(ArrayAngles, numOfQubits, MPI::DOUBLE, j, i);
                allAngles.col(runCount) = vec(ArrayAngles,numOfQubits);
                runCount++;
            }
        }
        
        //save the output in some file.
        allAngles.save("allAngles.txt",raw_ascii);
        
        //Convert to binary vectors and save as file as well. This file is numOfRuns rows, 8 columns. (Easier to read on Mac)
        imat allSpins = trans(conv_to<imat>::from(allAngles > datum::pi/2));
        allSpins.save("allSpins.txt",raw_ascii);
    }
   
    MPI::Finalize(); // clean up parallel process.
    return 0;
}
