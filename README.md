# MPI-Programming-Project
Aim of this project is implementing a parallel algorithm for feature selection using Relief. Parallel programs are needed beacuse of the large datasets. In our project there will be N processors. 1 master processor and N-1 slave processors. Master processos handles with I/O and sends these informations to slave processors. Slave Processors makes releated calculations and the can only print to console. Also, feture selection is made with Relief algorithm. Relief is a feature weighting algorithm which estimates the quality of attributes considering the strong dependencies between them. In order to do feature selection, weights of the features are calculated and the best weights are selected. 