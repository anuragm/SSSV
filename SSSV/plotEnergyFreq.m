%This file would plot the energies-frequency histogram for the
%hamiltonian. data is assumed to recide in /data/ folder.

[h,J,listOfQubits] = readHamiltonian('hamiltonian.config');

%In this case, we are only considering what happens for alpha=1.
%Load the file, and read all the states.

states = load('data/spins1.000.dat','-ascii');

numOfRuns = size(states,1);
energies = zeros(1,numOfRuns);

for ii=1:numOfRuns
    stateVector = -2*states(ii,:)'+1;
    energies(ii) = h*stateVector + stateVector'*J*stateVector;
end

[uniqueEnergies,~,occuranceLocation] = unique(energies); %Find the unique vectors, and bucket them. EigenVectors are row vectors
uniqueFreq = accumarray(occuranceLocation,1)';

%% Now plot
figure;
bar(uniqueEnergies,uniqueFreq/numOfRuns);
title('Hamiltonian 8 SSSV','Interpreter','Latex','FontSize',20)
xlabel('Energies','Interpreter','Latex','FontSize',18);
ylabel('Fractional Frequencies','Interpreter','Latex','FontSize',18);

set(gcf, 'Color', 'w'); %Set the background to white
set(gca,'FontSize',16,'FontName','Times New Roman');

%Standard size and dimensions for figure

set(gcf,'Position',[921   469   620   454]);
set(gcf,'PaperPositionMode','Auto','Units','Inches','PaperSize',[9 6.5]);
set(gcf, 'PaperPositionMode','Auto','Units','Inches', 'Position',[0 0 9 6.5]);