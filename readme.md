# Headers and Source

All algorithms headers are on the include folder, 
there are four header files, 

1. Data.h responsible to read training and test data 
2. Matrix.h is responsible to represent (user,item, rating) as a sparse matrix 
3. Model.h has the description of all CF methods
Thread.h is responsible for distributing work between threads

All the source files are in the scr folder there are eight .cpp file
1. Data.cpp implements functions to manipulate input/output
2. ItemBasedCF.cpp implementation of Item Based CF
3. Main.cpp runs the program
4. Matrix.cpp implements sparse matrix manipulation
5. MetaModel.cpp  meta model implementation
6. Svd.cpp singular value decomposition implementation
7. UserBasedCF.cpp implementation of user based CF
8. Validate.cpp implementation of function to predict ratings on test set

The makefile is also on the src folder

# Compilation:
To compile the source, first go to the folder src and next run the command make. 

# Execution: 

To execute just run on the src folder:

./recomender trainPath testPath 

the result is written to standard output. The default algorithm for execution
is the meta model, it can be changed by manually seting a different algorithm
in the Main.cpp and compiling the code.


# Observations
The meta algorithm output is on the data folder with the name meta.csv
