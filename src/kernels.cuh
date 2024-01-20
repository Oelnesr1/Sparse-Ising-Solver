#pragma once

// general kernels

__global__ void Update_XY(double *X, double *Y, int* step, int max_steps, double time_step, int ncols, int nrows);
__global__ void update_window(double *X, int *current_spins, int *final_spins, bool *bifurcated, bool *prev_bifurcated, int *stability, int convergence_threshold, int ncols, int nrows);

// MAXCUT-specific kernels

__global__ void mat_mult(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows);
__global__ void step_forward(int* step);
__global__ void confine(double *X, double *Y, int ncols, int nrows);

__global__ void mat_mult_and_confine(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows);
__global__ void symplectic_kernel_maxcut(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows);
__global__ void symplectic_kernel_maxcut_shared_memory(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows);
__global__ void compute_max_cut(int *a_pointers, int *a_indices, double *a_values, int *final_spins, double *cut_array, int ncols, int nrows);
__global__ void sum_max_cut(double *cut_array, double *final_cut_array, double sum, int ncols, int nrows);

// Traveling Salesman Problem (TSP)-specific kernels

__global__ void TSP_distances_GEO(double* distances, double *latitudes, double*longitudes, int ncities);
__global__ void TSP_distances_EUC_2D(double* distances, double *latitudes, double*longitudes, int ncities);
__global__ void tsp_spin_average(int *spin_array, double* average_array, int ncities, int ncols, int nrows);
__global__ void update_window_tsp(double *X, int *current_spins, int *final_spins, bool *bifurcated, bool *prev_bifurcated, int* stability, int* valid_list, int convergence_threshold, int* city_visits, int* order_visits, int* step, int ncities, int ncols, int nrows);
__global__ void symplectic_kernel_tsp(int *a_pointers, int *a_indices, double *a_values, double *total_distances, double *X, double *Y, double time_step, int* step, int max_steps, double xi0, int n, double A, double B, double C, int ncities, int ncols, int nrows);
