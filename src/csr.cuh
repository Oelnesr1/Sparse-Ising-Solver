#pragma once

#include <bits/stdc++.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <memory>
#include <utility>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <algorithm>
#include <random>

#include <string>
using std::string;

#include <vector>
using std::vector;

class CSR {
public:
    CSR(string filename);
    CSR(vector<double> matrix, int dim);
    vector<double> csr_data;
    vector<int> csr_row_indices;
    vector<int> csr_columns;

    int size;
    double sum = 0.0f;
    int nonzeros;

    void from_matrix(vector<double> matrix, int dim);
    void from_file(string filename);
    void construct_csr(vector<int> rows, vector<int> cols, vector<double> vals);

    vector<double> get_row(int row);
    vector<int> get_col_of_row(int row);

    vector<double> get_col(int row);
    vector<int> get_row_of_col(int row);
};

class TSP_CPU {
public:
    TSP_CPU(string filename);
    void read_file(string filename, int cities_to_read);

    int ncities;
    int nonzeros;
    bool override_ncities = false;
    int desired_cities = 8;

    string dtype;

    vector<double> latitudes;
    vector<double> longitudes;  
};

class TSP_GPU {
public:
    TSP_GPU(TSP_CPU& tsp_cpu);
    void initialize(TSP_CPU& tsp_cpu);
    vector<double> construct_matrix();
    void print_h();

    double maximum = 0;
    int ncities;
    int ndistances;
    double sum_distance = 0;

    double* total_distances_cpu;
    double* total_distances_gpu;
    double* gpu_latitudes;
    double* gpu_longitudes;
    double* gpu_distances;
    double* cpu_distances;

    double A;
    double B;
    double C;
};

class CSR_GPU {
public:
    CSR_GPU(CSR& csr_cpu);
    int* d_row_indices;
    int* d_columns;
    double* d_data;
};

class Ising {
public:
    Ising(CSR& csr_cpu, CSR_GPU &csr_gpu, double time_step = 0.1, int agents = 1000);
    ~Ising();
    void initialize(CSR& csr_cpu, CSR_GPU &csr_gpu, double param_time_step, int agents);
    void reset_initialization(CSR& csr_cpu, CSR_GPU &csr_gpu);
    double symplectic_update(CSR& csr_cpu, CSR_GPU& csr_gpu, std::ofstream& file, bool uw, bool fused, int max_steps = 10000);
    void generate_random(CSR& csr_cpu, CSR_GPU& csr_gpu);
    
private:
// parameters
    int convergence_threshold = 25;
    int sampling_period = 100;
    int max_steps = 10000;
    // int agents = 1000;
    double time_step;
    double pressure_slope = 0.01;
    
    int nrows;
    int ncols;
    int SIZE;

    double* cut_array_gpu;
    double* final_cut_array_cpu;
    double* final_cut_array_gpu;

    int* step_gpu;

    double* X_cpu;
    double* Y_cpu;

    double* X_gpu;
    double* Y_gpu;

    int* current_spins_gpu;
    int* final_spins_gpu;

    int* stability_gpu;

    bool* bifurcated_cpu;

    bool* bifurcated_gpu;
    bool* prev_bifurcated_gpu;

    bool run = true;
    int step = 0;

    double xi0;

    double pressure;

    size_t double_bytes;
    size_t double_bytes_large;
    size_t int_bytes_large;
    size_t int_bytes_small;
    size_t bool_bytes;
    size_t int_bytes_ncols;
};

class Ising_TSP {
public:
    Ising_TSP(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu);
    void initialize(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu);
    void reset_initialization(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu);
    double symplectic_update(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu, std::ofstream& file, int max_steps = 10000);
    void generate_random(TSP_CPU& tsp_cpu, TSP_GPU &tsp_gpu);
    double calculate_distance_of_tour(TSP_GPU &tsp_gpu, vector<int> cities, int ncities);
    
private:
// parameters
    int convergence_threshold = 50;
    int sampling_period = 25;
    int max_steps = 10000;
    int agents = 10000;
    double time_step = 0.1;

    int attempt = 1;

    double A;
    double B;
    double C;

    int ncities;
    int nrows;
    int ncols;
    int SIZE;

    bool run = true;
    int step = 0;

    double xi0;
    bool optimal_xi0 = true;

// cpu pointers

    double* X_cpu;
    double* Y_cpu;

    int* current_spins_cpu;
    int* final_spins_cpu;
    double* average_spin_cpu;

    int* valid_list_cpu;

// gpu pointers

    double* X_gpu;
    double* Y_gpu;

    int* current_spins_gpu;
    int* final_spins_gpu;
    double* average_spin_gpu;

    int* stability_gpu;
    bool* bifurcated_gpu;
    bool* prev_bifurcated_gpu;

    int* valid_list_gpu;
    int* city_visits_gpu;
    int* order_visits_gpu;

    int* step_gpu;


// memory allocation sizes

    size_t double_bytes_SIZE;
    size_t int_bytes_SIZE;
    size_t int_bytes_nrows;
    size_t bool_bytes_ncols;
};


