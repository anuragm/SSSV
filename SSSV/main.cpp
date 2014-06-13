//  main.cpp
//  SSSV
//
//  Created by Anurag Mishra on 5/14/14.
//  Copyright (c) 2014 Anurag Mishra. All rights reserved.

#include <iostream>
#include <armadillo>
#include <ctime>
#include <mpi.h>

#include "runSSSV.hpp"

#define MASTER 0    // defines the master thread, or the root node.

int main(int argc, char * argv[])
{
    //Run alpha values from configuration file scaling.config
    arma::vec alpha = getScalings();
    
    MPI::Init(argc,argv);                          //Initialize openMPI
    int numOfThreads = MPI::COMM_WORLD.Get_size(); //Tells the total number of thread availible.
    int node_id      = MPI::COMM_WORLD.Get_rank(); //Gives the id of the current thread
    
    //Initialize h and J from reading disk.
    arma::vec h_noNoise; arma::mat J_noNoise;
    readHamiltonian(&h_noNoise, &J_noNoise); //reads the values from hamiltonian.config
    int numOfQubits = h_noNoise.n_elem; //identify total number of qubits in the simulation.
    
    //Common parameters for all runs
    int numOfSSSVRuns;
    int numOfSweeps;
    double temperature;
    double noise; //gives the standard deviation of the noise to be used.
    readParameters(&numOfSSSVRuns, &numOfSweeps, &temperature, &noise); //reads parameters from SSSV.config.
    
    arma::mat dw2schedule;
    dw2schedule.load("dw2schedule.txt",arma::raw_ascii); //Loads the required schedule.
    
    for(int c_alpha=0;c_alpha<alpha.n_elem;c_alpha++) //loop over alpha
    {
        //initialize random number generator differently for each node, and for each alpha
        long int seed = std::time(NULL) + node_id;
        srand48(seed);
        
        //Num of jobs to be done by current thread.
        int numOfJobs = numOfSSSVRuns/numOfThreads + ((node_id<numOfSSSVRuns%numOfThreads)?1:0) ;
        
        //Auxillary threads broadcast their calculations.
        if(node_id!=MASTER)
        {
            //run each job one by one, and send the result to master node.
            for (int iiRuns=0; iiRuns < numOfJobs;iiRuns++)
            {
                arma::vec h; arma::mat J;
                addNoise(&h, &J, h_noNoise*alpha(c_alpha), J_noNoise*alpha(c_alpha), noise);
                arma::vec VecAngles = runSSSV(h,J,numOfSweeps,temperature,dw2schedule);
                
                //convert vector to an double array of size numOfQubits
                double ArrayAngles[numOfQubits];
                std::memcpy(ArrayAngles, VecAngles.memptr(), numOfQubits*sizeof(double));
                
                //send the resultant array to master node
                int jobTag = iiRuns + numOfSSSVRuns*c_alpha;
                MPI::COMM_WORLD.Send(ArrayAngles,numOfQubits,MPI::DOUBLE,MASTER,jobTag);
            }
        } //end auxillary node work
        
        //Master node collects the data, and saves them to disk.
        if(node_id==MASTER)
        {
            std::cout<<" working on alpha="<<alpha(c_alpha)<<std::endl;
            //Run master nodes jobs.
            arma::mat allAngles(numOfQubits,numOfSSSVRuns);
            int runCount =0;
            for(int c_jobs=0;c_jobs<numOfJobs;c_jobs++)
            {
                arma::vec h; arma::mat J;
                addNoise(&h, &J, h_noNoise*alpha(c_alpha), J_noNoise*alpha(c_alpha), noise);
                std::cout<<"Master node got a copy of Hamiltonian, now running it."<<std::endl;
                allAngles.col(runCount)=runSSSV(h, J, numOfSweeps, temperature,dw2schedule);
                std::cout<<"Run done"<<std::endl;
                runCount++;
            }
            std::cout<<"Master node is done running its own jobs."<<std::endl;
            //Receive jobs from other nodes.
            for(int c_nodes=1;c_nodes<numOfThreads;c_nodes++)
            {
                //calculate number of jobs done by that node.
                int j_NumOfJobs = numOfSSSVRuns/numOfThreads + ((c_nodes<numOfSSSVRuns%numOfThreads)?1:0);
                //receive all data send from thread j, and convert it into a column of allAngles
                for(int c_jobs=0;c_jobs<j_NumOfJobs;c_jobs++)
                {
                    double ArrayAngles[numOfQubits];
                    int jobTag = numOfSSSVRuns*c_alpha+c_jobs;
                    std::cout<<"Master node is waiting to recieve jobTag "<<jobTag<<std::endl;
                    MPI::COMM_WORLD.Recv(ArrayAngles, numOfQubits, MPI::DOUBLE, c_nodes, jobTag);
                    allAngles.col(runCount) = arma::vec(ArrayAngles,numOfQubits);
                    std::cout<<"Master node recieved jobTag "<<jobTag<<" and has saved it to array "<<std::endl;
                    runCount++;
                }
            }
            //save the output in file.
            std::ostringstream fileToSave;
            fileToSave.precision(3);
            fileToSave.setf( std::ios::fixed, std::ios::floatfield ); //Pad with zeros if required.
            fileToSave<<"angles"<<alpha(c_alpha)<<".dat";
            allAngles.save(fileToSave.str(),arma::raw_ascii);
            
            //Convert to binary vectors and save as file as well. This file is numOfRuns rows, numOfQubit columns.
            arma::imat allSpins = trans(arma::conv_to<arma::imat>::from(allAngles > arma::datum::pi/2));
            fileToSave.clear(); fileToSave.str("");
            fileToSave<<"spins"<<alpha(c_alpha)<<".dat";
            allSpins.save(fileToSave.str(),arma::raw_ascii);
            
        } //end MASTER work
    } //end of alpha loop.
    
    MPI::Finalize(); // clean up parallel process.
    return 0;
}
