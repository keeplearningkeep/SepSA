# SepSA
Our algorithm is implemented using the pytorch framework, where the "slimtik_functions" file are the functions used to solve the parameters of the final layer in the 
code provided by the slimTrain paper. Run the python file prefixed with "run" to start the corresponding experiment, the "online" and "minibatch" suffixes correspond
to the experiments of online learning and mini-batch learning respectively. The experimental results shown in our paper are the results when the seed value is 233. 
The results of the regression datasets in our paper were run on the CPU, and the results of the classification data set were run on the GPU. The time calculation of the experiment 
of taking the average of multiple different seeds does not perform any calculation of training error and test error during the training process.
