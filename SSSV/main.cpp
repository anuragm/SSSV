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
    
    //Initialize h and J differently for each thread.
    vec h(8); mat J(8,8);
    double noise=0.085; //gives the standard deviation of the noise to be used.
    
    //Common parameters for all runs
    int NumOfSSSVRuns  = 10000;  //Number of times SSSV should be run.
    int numOfSweeps    = 150;
    double temperature = 1.383; //Temperature used by Shin et al
    
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
    
    for(int c_alpha=0;c_alpha<alpha.n_elem;c_alpha++) //loop over alpha
    {
        //initialize random number generator differently for each node, and for each alpha
        long int seed = time(NULL) + node_id;
        srand48(seed);
        
        //Num of jobs to be done by current thread.
        int numOfJobs = NumOfSSSVRuns/numOfThreads + ((node_id<NumOfSSSVRuns%numOfThreads)?1:0) ;
        
        //Auxillary threads broadcast their calculations.
        if(node_id!=MASTER)
        {
            //run each job one by one, and send the result to master node.
            for (int iiRuns=0; iiRuns < numOfJobs;iiRuns++)
            {
                h.zeros(); J.zeros(); //Initialize all the unused couplings to zero.
                getSigHam(alpha(c_alpha), noise*noise, &h, &J); //reinitialize couplings before every run.
                vec VecAngles = runSSSV(-h,-J,numOfSweeps,temperature,dw2schedule);
                
                //convert vector to an double array of size numOfQubits
                double ArrayAngles[numOfQubits];
                memcpy(ArrayAngles, VecAngles.memptr(), numOfQubits*sizeof(double));
                
                //send the resultant array to master node
                int jobTag = iiRuns + NumOfSSSVRuns*c_alpha;
                MPI::COMM_WORLD.Send(ArrayAngles,numOfQubits,MPI::DOUBLE,MASTER,jobTag);
            }
        } //end auxillary node work
        
        //Master node collects the data, and saves them to disk.
        if(node_id==MASTER)
        {
            //Run master nodes jobs.
            mat allAngles(numOfQubits,NumOfSSSVRuns);
            int runCount =0;
            for(int c_jobs=0;c_jobs<numOfJobs;c_jobs++)
            {
                h.zeros(); J.zeros(); //Initialize all the unused couplings to zero.
                getSigHam(alpha(c_alpha), noise*noise, &h, &J); //reinitialize couplings before every run.
                allAngles.col(runCount)=runSSSV(-h, -J, numOfSweeps, temperature,dw2schedule);
                runCount++;
            }
            //Receive jobs from other nodes.
            for(int c_nodes=1;c_nodes<numOfThreads;c_nodes++)
            {
                //calculate number of jobs done by that node.
                int j_NumOfJobs = NumOfSSSVRuns/numOfThreads + ((c_nodes<NumOfSSSVRuns%numOfThreads)?1:0);
                //receive all data send from thread j, and convert it into a column of allAngles
                for(int c_jobs=0;c_jobs<j_NumOfJobs;c_jobs++)
                {
                    double ArrayAngles[numOfQubits];
                    int jobTag = NumOfSSSVRuns*c_alpha+c_jobs;
                    MPI::COMM_WORLD.Recv(ArrayAngles, numOfQubits, MPI::DOUBLE, c_nodes, jobTag);
                    allAngles.col(runCount) = vec(ArrayAngles,numOfQubits);
                    runCount++;
                }
            }
            //save the output in file.
            ostringstream fileToSave;
            fileToSave.precision(3);
            fileToSave.setf( ios::fixed, ios::floatfield ); //Pad with zeros if required.
            fileToSave<<"angles"<<alpha(c_alpha)<<".dat";
            allAngles.save(fileToSave.str(),raw_ascii);
            
            //Convert to binary vectors and save as file as well. This file is numOfRuns rows, 8 columns. (Easier to read on Mac)
            imat allSpins = trans(conv_to<imat>::from(allAngles > datum::pi/2));
            fileToSave.clear(); fileToSave.str("");
            fileToSave<<"spins"<<alpha(c_alpha)<<".dat";
            allSpins.save(fileToSave.str(),raw_ascii);
            
        } //end MASTER work
    } //end of alpha loop.
    
    MPI::Finalize(); // clean up parallel process.
    return 0;
}
