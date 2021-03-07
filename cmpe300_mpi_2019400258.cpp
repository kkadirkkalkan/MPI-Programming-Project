//Student Name: Kadir Kalkan
//Student Number: 20019400258
//Compile Status: Compiling
//Program Status: Working

#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <set>
#include <cmath>
using namespace std;
string inputPath; // holds name of the input file which is in the same directory with code
double findweight(double instinfeat, double hitinfeat, double missinfeat, double maxfeat, double minfeat, int M ); // makes calculations in order to calculate weight
double manhattan(double arr1[], double arr2[], int A ); // finds manhattan distance between two arrays also function takes A as a paramater. A equals arraysize-1.
void Insert2DArray(double** subarray, int xindex, int yindex, double m); //This function inserts a value in to this 2D array
void Initilaze2DArray(double*** subarray, int sizex, int sizey); //This function initilazes a 2D array

int main(int argc, char *argv[]) {
    int rank; //holds the id of the current processor
    int size; //holds the total number of the processors
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;


    int word; //to hold the value that is read from input file
    int N; //shows the total number of processors
    int P; //shows the total number of instances
    int A; //shows total number of features
    int M; //Shows the iteration count
    int T; //shows resulting number of features

    inputPath = argv[1]; //I toke the name of the input file from command line with argv[1]
    ifstream file;


    if (rank == 0) {  //the master process reads these 5 values from input file
        file.open(inputPath);

        file >> word;
        N = word;
        file >> word;
        P = word;
        file >> word;
        A = word;
        file >> word;
        M = word;
        file >> word;
        T = word;

    }


        MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);//Master sents P to slaves
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);//Master sents N to slaves
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD); //Master sents M to slaves
        MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);//Master sents T to slaves
        MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);//Master sents A to slaves




    long rk = ((P / (N - 1)) * (A + 1)); //the range of the “pref ” array

    long rs = rk * N; //the range of the “arr” array



    double *arr = new double[rs]; // holds the all data in the input file and zero values for master process.
    double *pref = new double[rk]; // holds the datas that will be sent to each processors

    for (int m = 0; m <rk ; ++m) { //initialize the array’s values to zero
        pref[m]=0;
    }

    for (int n = 0; n <rs ; ++n) { //initialize the array’s values to zero
        arr[n]=0;
    }

    if (rank == 0) { //master process reads the datas from input file and send them to the array “arr”


        double num;
        long j = 0, i = rk;
        for (; j < rk; j++)  //The first rk values of “arr” array equas 0 for master process.
            arr[j] = 0;
        while (file >> num) {  // reding datas from input file
            arr[i] = num;
            i++;
        }
        file.close();




    }
    MPI_Scatter(arr, rk, MPI_DOUBLE, pref, rk, MPI_DOUBLE, 0, MPI_COMM_WORLD); //MPI_Scatter sends the data in the array “arr” to “pref” array that special for all processes


   long *bestweigths = new long[T]; //holds the index of the best T weights for each slaves

    if (rank != 0) {


        int numberofarray = P / (N - 1); //holds the number of instances for one process
        if(M>numberofarray){ //The given iteration number must be equal or smaller than this value
            M=numberofarray;
        }


        double** subarray = NULL;
        int xsize=numberofarray, ysize=A+1; //column size equals instance number and row size equals A+1
        Initilaze2DArray(&subarray, xsize, ysize); //I created a 2D array from “pref” array for each slave process


        for (int m = 0; m <numberofarray ; ++m) { //insert all values in “pref” array to “subarray”
            for (int i = 0; i <A+1 ; ++i) {
                Insert2DArray(subarray, m,i, pref[(m*(A+1))+i]);
            }

        }





        double *maxfeat = new double[A]; // holds the maximum values in each features
        double *minfeat = new double[A]; // hods the minimum values in each features

    for (long col = 0; col < A; col++) {
        double maxvalue; // max value in a feature
        int max1 = 0; // in order to assign first instance of the each feature to “maxvalue” and “minvalue”

        double minvalue; // min value in a feature
        for (long row = 0; row < numberofarray; row++) { //if the other iterations of this feature greater than the “maxvalue” the new “maxvalue” changes and if the other iterations of this feature smaller than the “minvalue” the new “minvalue” changes.
            if (max1 == 0) {
                maxvalue = subarray[row][col];
                minvalue = subarray[row][col];
                max1++;
            } else if (subarray[row][col] > maxvalue) {
                maxvalue = subarray[row][col];
            } else if (subarray[row][col] < minvalue) {
                minvalue = subarray[row][col];
            }

        }
        maxfeat[col] = maxvalue;
        minfeat[col] = minvalue;
    }


    long iterationindex; //holds the instance index of the target instances



    double *weightarray = new double[A];  // this array holds weights of the features
    for (long i = 0; i < A; i++) {   // at first initilaze all waights to zero
        weightarray[i] = 0;
    }
    for (iterationindex = 0; iterationindex < M; iterationindex++) { //I found the nearestmiss and nearesthit values in this part

        double nearesthit; // holds the smallest manhattan distance value between target instance and an instance that has the same class label with target instance
        double nearestmiss; //holds the smallest manhattan distance value between target instance and an instance that has the different class label with target instance
        int hit1 = 0;
        int miss1 = 0;
        long nearesthitindex = -1;
        long nearestmissindex = -1;

        long iterationclass = subarray[iterationindex][A]; //holds the class of the iteration
        for (long s = 0; s < numberofarray; s++) {  // search all arrays to find nearest hit and nearest miss

            if (s != iterationindex) {    // searching shouldn't hold for the target instance
                if (iterationclass == subarray[s][A]) {  //search for the nearest hit
                    double currenthit = 0;
                    currenthit = manhattan(subarray[s], subarray[iterationindex], A);

                    if (hit1 == 0) {  //Firstly, hit1 is zere so the first hit value will be “nearesthit” .
                        nearesthit = currenthit;
                        nearesthitindex = s;
                        hit1++;
                    } else if (currenthit < nearesthit) {
                        nearesthit = currenthit;
                        nearesthitindex = s;

                    }
                } else if (iterationclass != subarray[s][A]) { // searchs for the nearest miss


                    double current = 0;
                    current = manhattan(subarray[s], subarray[iterationindex], A);

                    if (miss1 == 0) { ////Firstly, miss1 is zere so the first miss value will be “nearesthit” .
                        nearestmiss = current;
                        nearestmissindex = s;
                        miss1++;
                    }

                    if (current < nearestmiss) {

                        nearestmiss = current;
                        nearestmissindex = s;

                    }

                }
            }
        }

        for (long featnum = 0; featnum < A; featnum++) { //In order to create “weightarray”, for each feature, the target instance value, hit value, miss value, max value, min value and M value is sent to findweight function.



            weightarray[featnum] = weightarray[featnum] +
                                   findweight(subarray[iterationindex][featnum], subarray[nearesthitindex][featnum],
                                              subarray[nearestmissindex][featnum], maxfeat[featnum],
                                              minfeat[featnum], M);


        }

    }




    double *unsortedweight = new double[A]; //its values are the same with “weightarray”

    for (long i = 0; i < A; i++) {

        unsortedweight[i] = weightarray[i];
    }


    sort(weightarray, weightarray + A,greater<double>());  // sorting the weight array in descenging order to take best T weights
    for (long i = 0; i < T; i++) { //first T weights in the “weightarray” array

        for (long k = 0; k < A; k++) { //for all values in the “unsortedweight” array
            if (weightarray[i] == unsortedweight[k]) { //If the value in the “weightarray” equals the value in the “unsortedweight” the index value of “unsortedweight” array is the one of the best T features’s indexes.
                bestweigths[i] = k;

            }
        }
    }


    sort(bestweigths, bestweigths + T); //sorting "bestweights" in order to output order




    cout<<"Slave P"<<to_string(rank)<<" :";

    for(long i=0 ; i<T ; i++){

        cout<<" "<<bestweigths[i];
    }
    cout<<endl;


    MPI_Send(bestweigths,T,MPI_LONG,0,1,MPI_COMM_WORLD); //I sent "bestweights" to master process for all slave processes with MPI_Send


   delete [] subarray;
   delete [] pref;
   delete [] maxfeat;
   delete [] minfeat;
   delete [] weightarray;
   delete [] unsortedweight;
}





    if(rank==0){

        set<long> s; // set that holds the best indexes with no duplicate elements.
        for (long k = 1; k <N ; k++) {  //master process receives these arrays with MPI_Recv from all slave processes
            MPI_Recv (bestweigths,T,MPI_LONG,k,1,MPI_COMM_WORLD,&status);
            for (long i = 0; i <T ; i++) {

                s.insert(bestweigths[i]); //insert the values in the “bestweights” array to “s” set


            }
        }

        cout<<"Master P0 :";

        set<long>::iterator it;
        for (it = s.begin(); it != s.end(); ++it) {
            cout << " " <<*it;
        }
        cout<<endl;
        delete [] bestweigths;
    }


    MPI_Barrier(MPI_COMM_WORLD); // synchronizing processes
    MPI_Finalize();

    delete [] arr;
    return 0;
}



double manhattan(double arr1[], double arr2[], int A){   //gets two array and arraysize-1 and returns the manhattan distances between two array (don't include the last element of the array beacuse this value is the class)

    double count=0;

    for(long i=0; i<A; i++){
        double fark=arr1[i]-arr2[i];

        if(fark<0){
            count=count-fark;
        }
        else{
            count=count+fark;
        }

    }

    return count;
}

double findweight(double instinfeat, double hitinfeat, double missinfeat, double maxfeat, double minfeat, int M ){ //The return value equals – (| targetinstance value – hit value| /(maxvalue-minvalue)) /M +  (| targetinstance value – miss value| /(maxvalue-minvalue)) /M

    double result=0;
    result=result- (((abs(instinfeat-hitinfeat))/(maxfeat-minfeat))/M);

    result=result+ (((abs(instinfeat-missinfeat))/(maxfeat-minfeat))/M);



    return result;

}

void Insert2DArray(double** subarray, int xindex, int yindex, double m)
{

    (subarray)[xindex][yindex] = m;    // inserr value m to subarray with column xindex and row yindex

}

void Initilaze2DArray(double*** subarray, int sizex, int sizey) // creating 2D array with column size sizex and row size sizey.
{

    *subarray = new double*[sizex];
    for (int i=0;i<sizex;i++)
    {
        (*subarray)[i] = new double[sizey];
    }
}