function [h,J,listOfQubits] = readHamiltonian(fileName)
% READHAMILTONIAN - reads hamiltonain from DW2 compatible file to two
% vectors
% Input : fileName - the name of file which stores hamiltonain in DW2
% format
% Output: h - local fields
%         J - coupling matrix
%         listOfQubits - vector of functional qubits in Hamiltonian.

fileHandle = fopen(fileName);

firstLine = fgets(fileHandle);
dataRead  = sscanf(firstLine,'%d %d');

numOfQubits = dataRead(1); numOfLines = dataRead(2);

h = zeros(1,numOfQubits);
J = zeros(numOfQubits,numOfQubits);
listOfQubits = [];

for ii=1:numOfLines
    lineRead = fgets(fileHandle);
    dataRead = sscanf(lineRead,'%d %d %f');
    
    index1 = dataRead(1); index2 = dataRead(2); value = dataRead(3);
    
    if index1==index2
        h(index1+1) = value; %+1 to convert to Matlab indexing
    elseif index1>index2
        J(index2+1,index1+1) = value;
    else
        J(index1+1,index2+1) = value;
    end
    
    listOfQubits = [listOfQubits index1 index2];
    
end

listOfQubits = unique(listOfQubits);
end