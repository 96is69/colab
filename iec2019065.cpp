/******************************************************************************

ROLL NUMBER : IEC2019065
NAME        : JAYA MUKESH G

DESCRIPTION : Program to multipl sparse matrix and Dense Vector with and without using threads

SUMMARY     : First the COO file is read and converted into CSR format with matrix_to_csr() function.
              Then matrix-vector multiplication is performed using threads by thread_multiplication() 
              function and result is printed. Then matrix-vector multiplication is performed without 
              using threads by multiplication() function and result is printed.

OUTPUT      :   1) A-vector in CSR and 2d formats.
                2) C-vector and time taken by each method

OS          : Ubuntu
COMPILER    : g++
COMMAND     : g++ -std=c++11 -pthread iec2019065.cpp

*******************************************************************************/

#include<bits/stdc++.h>
using namespace std;
using namespace std::chrono;

// thread parameters
int num_of_threads;
int thread_count_track =0;

// Matrix A components for CSR format
// vector<double>csr_val, csr_row, csr_col;
double *csr_row, *csr_col, *csr_val;

//Matrix B and C
// vector<double>b_vector;
// vector<double>c_vector;
double **b_vector, **c_vector;
int vector_size;

double **matrix;


// function to convert 2d sparse matrix to CSR format
void matrix_to_csr(vector<vector<double>>&matrix)
{
    int temp_count = 0;

    for(int i=0;i<matrix.size();i++)
    {
        for(int j=0;j<matrix[i].size();j++)
        {
            if(matrix[i][j] != 0)
            {
                csr_val.push_back(matrix[i][j]);
                csr_col.push_back(j);
                temp_count++;
            }
        }

        csr_row.push_back(temp_count);
    }
}

// function to perform CSR-vector multiplication without using threads
void multiplication()
{
    for(int i=0;i<vector_size;i++)
    {
        int position = 0;
        if(i) position = csr_row[i-1];
        for(int j=position; j<csr_row[i] ; j++)
        {
            c_vector[i] += csr_val[j] * b_vector[csr_col[j]];
        }
    }
}

// function to perform CSR-vector multiplication by using threads
void thread_multiplication()
{
    int position = 0;
    // find the batch to be processed by the current thread
    int batch_start = (vector_size/num_of_threads)*thread_count_track;
    int batch_end = (vector_size/num_of_threads)*(thread_count_track+1);

    thread_count_track += 1;

    for( int i = batch_start; i<= batch_end && i<vector_size; i++ )
    {
        int position = 0;
        if(i) position = csr_row[i-1];
        for(int j=position; j<csr_row[i] ; j++)
        {
            c_vector[i] += csr_val[j] * b_vector[csr_col[j]];
        }
    }
}

void allocate_matrix()
{
  matrix = malloc((M*N)*sizeof(double));
}

int main()
{
    // read COO file
    ifstream file("inputfile.mtx");
    int M, N, L;
    while (file.peek() == '%') file.ignore(2048, '\n');
    file >> M>> N >> L;

    vector_size = M;

        // 2d matrix declaration
    // vector<vector<double>> matrix(M, vector<double>(N, 0.0));
    allocate_matrix();
    
        // fill the matrix with non zero values while reading COO file
    for (int l = 0; l < L; l++)
    {
        double data;
        int csr_row, csr_col;
        file >> csr_row >> csr_col >> data;
        matrix[csr_row -1][csr_col -1] = data;
    }
        
    file.close();
    
    // read vector.txt file
    fstream fn;
    string word, filename;
    filename = "vector.txt";
    fn.open(filename.c_str());
    while (fn >> word)
    {
        int x = stoi(word);
        b_vector.push_back(x);
    }
    
    // covert 2d matrix to CSR format
    matrix_to_csr(matrix);
    
    cout<<"\nCSR Value vector: \n";
    for(auto i:csr_val) cout<<i<<" "; cout<<"\n";
    cout<<"\nCSR Column vector: \n";
    for(auto i:csr_col) cout<<i<<" "; cout<<"\n";
    cout<<"\nCSR Row vector: \n";
    for(auto i:csr_row) cout<<i<<" "; cout<<"\n";

    // print matrix A
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<matrix[i][j]<<" ";
        }
        cout<<endl;
    }

    cout<<endl;
    
    
    // multiplication with using threads
    cout<<"\nEnter Number of threads to use: ";
    cin>>num_of_threads;
    cout<<"\nMultiplication Using Threads\n";
    
    vector<thread> threads(num_of_threads);
    c_vector.assign(vector_size, 0);

        // START TIMER
    auto start = high_resolution_clock::now();

    for(int i=0;i<num_of_threads;i++)
    threads[i] = thread(thread_multiplication);

    for(int i=0;i<num_of_threads;i++)
    threads[i].join();

        // STOP TIMER AND PRINT DURATION
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<< "\nTime taken : " << duration.count() << " microseconds\n" << endl;

        // PRINT C_VECTOR
    for(auto i:c_vector) 
    cout<<i<<" "; 
    cout<<"\n";
    
    // multiplication without using threads
    cout<<"\nMultiplication Without Using Threads : \n";
    
    
    c_vector.assign(vector_size, 0);

        // START TIMER
    start = high_resolution_clock::now();
    multiplication();
        // STOP TIMER AND PRINT DURATION
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<< "\nTime taken : " << duration.count() << " microseconds\n" << endl;


        // PRINT C_VECTOR
    for(auto i:c_vector) 
    cout<<i<<" "; 
    cout<<"\n";
    
    return 0;
}
