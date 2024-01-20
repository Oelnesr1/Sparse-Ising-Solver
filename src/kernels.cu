// general kernels:

#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void Update_XY(double *X, double *Y, int* step, int max_steps, double time_step, int ncols, int nrows)
{
    // X represents the position of the oscillator
    // Y represents the momentum of the oscillator

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    double pressure = 1.025 * (double) step[0] / max_steps;

    if (ROW < nrows && COL < ncols)
    {
        Y[ROW * ncols + COL] += time_step * (pressure - 1) * X[ROW * ncols + COL];
        X[ROW * ncols + COL] += time_step * (pressure + 1) * Y[ROW * ncols + COL];
    }
}

__global__ void update_window(double *X, int *current_spins, int *final_spins, bool *bifurcated, bool *prev_bifurcated, int* stability, int convergence_threshold, int ncols, int nrows)
{
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = 0;
    bool equal;
    bool new_bifurcated;

    if (ROW == 0 && COL < ncols)
    {

        // for loop checks if current spin arrangement is identical to previous spin arrangement
        for (int i = 0; i < nrows; i++)
        {
            double val = X[ncols * i + COL];
            int x_sign = (signbit(val) * -2) + 1;
            
            // val / abs(val);
            sum += x_sign * current_spins[ncols * i + COL];
        }

        equal = (sum == nrows);

        bool not_equal = !equal;
        bool not_bifurcated = !bifurcated[COL];

        // if the oscillator has not bifurcated and has not changed, add 1 to the stability
        
        stability[COL] += equal && not_bifurcated;

        // if it has not bifurcated and has changed, reset the stability to 0; otherwise, do not change it

        stability[COL] = not_equal && not_bifurcated ? 0 : stability[COL];

        // if the stability has reached the convergence threshold, then the oscillator has bifurcated (converged)

        bifurcated[COL] = stability[COL] == convergence_threshold - 1;

        // if the oscillator has just bifurcated, then new_bifurcated is true

        new_bifurcated = bifurcated[COL] ^ prev_bifurcated[COL];

        // moves forward in time by moving the present bifurcation array into the previous bifurcation array

        prev_bifurcated[COL] = bifurcated[COL];

        // if the oscillator has bifurcated, we let the sign of the position of the oscillator become the final spins — should only happen once per oscillator

        if (new_bifurcated) {
            for (int i = 0; i < nrows; i++) {
                double val = X[i * ncols + COL];
                int x_sign = (signbit(val) * -2) + 1;
                final_spins[i * ncols + COL] = x_sign;
            }
        }
    }

    // we update our current spins every time this kernel is run, with the same algorithm as the final spins

    if (ROW < nrows && COL < ncols)
    {
        double val = X[ROW * ncols + COL];
        int x_sign = (signbit(val) * -2) + 1;
        // val / abs(val);
        current_spins[ROW * ncols + COL] = x_sign;
    }
}

// MAXCUT-specific kernels:

__global__ void mat_mult_and_confine(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows) {

    // the main algorithm

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    double total_interaction = 0;

    // calculation of pressure at the time step

    double pressure;
    // pressure = pressure_slope * time_step * step[0];
    pressure = 1.025 * (double) step[0] / max_steps;
    pressure = pressure < 1.0 ? pressure : 1.0;

    if (ROW < nrows && COL < ncols)
    {
        // enforce inelastic walls, by confining all positions to be within the range [-1, 1], and setting momenta to 0 if outside of boundary

        double val = X[ROW * ncols + COL];
        if (abs(val) > 1) 
        {
            X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
            // val / abs(val);
            Y[ROW * ncols + COL] = 0;
        }

        // sparse matrix multiplication: the sparse matrix encodes the interaction coefficients for each spin with all other spins, then calculates with matrix multiplcation
        else
        {
            int start_pointer = a_pointers[ROW];
            int end_pointer = a_pointers[ROW + 1];
            for (int i = start_pointer; i < end_pointer; i++)
            {
                double x_value = X[COL + ncols * a_indices[i]];
                int activated = (signbit(x_value) * -2) + 1;
                // (x_value / abs(x_value));
                total_interaction -= a_values[i] * activated;
            }

            // based on interactions, update the positions and momenta.
            Y[ROW * ncols + COL] += time_step * xi0 * total_interaction;
        }
        // 1: yyx
        // 2: y : yx
        // 3. yxy
        // 4. y : xy

        // update the step number once each time the kernel is run

        if (ROW == 0 && COL == 0) step[0] += 1;
    }
    
}
__global__ void confine(double *X, double *Y, int ncols, int nrows)
{

    // the main algorithm

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW < nrows && COL < ncols)
    {
        double val = X[ROW * ncols + COL];
        if (abs(val) > 1) {
                X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
                // val / abs(val);
                Y[ROW * ncols + COL] = 0;
        }
    } 
}

__global__ void step_forward(int* step)
{

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW == 0 && COL == 0) step[0] += 1;
}

__global__ void mat_mult(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows)
{

    // the main algorithm

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    double total_interaction = 0;

    // calculation of pressure at the time step

    double pressure;
    // pressure = pressure_slope * time_step * step[0];
    pressure = 1.025 * (double) step[0] / max_steps;
    pressure = pressure < 1.0 ? pressure : 1.0;

    if (ROW < nrows && COL < ncols)
    {
        int start_pointer = a_pointers[ROW];
        int end_pointer = a_pointers[ROW + 1];
        for (int i = start_pointer; i < end_pointer; i++)
        {
            double x_value = X[COL + ncols * a_indices[i]];
            int activated = (signbit(x_value) * -2) + 1;
            // (x_value / abs(x_value));
            total_interaction -= a_values[i] * activated;
        }

            // based on interactions, update the positions and momenta.
            Y[ROW * ncols + COL] += time_step * xi0 * total_interaction;
        }
}

__global__ void symplectic_kernel_maxcut(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows)
{

    // the main algorithm

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    double total_interaction = 0;

    // calculation of pressure at the time step

    double pressure;
    // pressure = pressure_slope * time_step * step[0];
    pressure = 1.025 * (double) step[0] / max_steps;
    pressure = pressure < 1.0 ? pressure : 1.0;

    if (ROW < nrows && COL < ncols)
    {
        // update the step number once each time the kernel is run

        if (ROW == 0 && COL == 0) step[0] += 1;

        // enforce inelastic walls, by confining all positions to be within the range [-1, 1], and setting momenta to 0 if outside of boundary

        double val = X[ROW * ncols + COL];
        if (abs(val) > 1) 
        {
            X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
            // val / abs(val);
            Y[ROW * ncols + COL] = 0;
        }

        // sparse matrix multiplication: the sparse matrix encodes the interaction coefficients for each spin with all other spins, then calculates with matrix multiplcation
        else
        {
            int start_pointer = a_pointers[ROW];
            int end_pointer = a_pointers[ROW + 1];
            for (int i = start_pointer; i < end_pointer; i++)
            {
                double x_value = X[COL + ncols * a_indices[i]];
                int activated = (signbit(x_value) * -2) + 1;
                // (x_value / abs(x_value));
                total_interaction -= a_values[i] * activated;
            }

            // based on interactions, update the positions and momenta.
            Y[ROW * ncols + COL] += time_step * xi0 * total_interaction;
        }

        X[ROW * ncols + COL] += time_step * (pressure + 1) * Y[ROW * ncols + COL];
        Y[ROW * ncols + COL] += time_step * (pressure - 1) * X[ROW * ncols + COL];
        // 1: yyx
        // 2: y : yx
        // 3. yxy
        // 4. y : xy
    }
    
}


__global__ void symplectic_kernel_maxcut_shared_memory(int *a_pointers, int *a_indices, double *a_values, double *X, double *Y, double pressure_slope, double time_step, int* step, int max_steps, double xi0, int ncols, int nrows)
{

    // the main algorithm

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + gridDim.y * blockIdx.z;

    if (ROW >= nrows) return;

    bool confined = false;
    double total_interaction = 0;

    // calculation of pressure at the time step

    double pressure;
    pressure = 1.025 * (double) step[0] / max_steps;
    pressure = pressure < 1.0 ? pressure : 1.0;

    // update the step number once each time the kernel is run

    if (ROW == 0 && COL == 0) step[0] += 1;

    // enforce inelastic walls, by confining all positions to be within the range [-1, 1], and setting momenta to 0 if outside of boundary

    if (COL < ncols) {
        double val = X[ROW * ncols + COL];
        if (abs(val) > 1)
        {
            X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
            // val / abs(val);
            Y[ROW * ncols + COL] = 0;
            confined = true;
        }
    }

    __shared__ float s_values[1024];
    __shared__ int s_indices[1024];

    int start_pointer = a_pointers[ROW];
    int end_pointer = a_pointers[ROW + 1];

    int number_nonzeros_in_row = end_pointer - start_pointer;

    // sparse matrix multiplication: the sparse matrix encodes the interaction coefficients for each spin with all other spins, then calculates with matrix multiplcation

    for (int j = 0; j < ceil((float)number_nonzeros_in_row / blockDim.x); j++)  {
        
        int idx = threadIdx.x + (blockDim.x * j);
        
        if (idx < number_nonzeros_in_row) {
            s_values[threadIdx.x] = a_values[start_pointer + idx];
            s_indices[threadIdx.x] = a_indices[start_pointer + idx];
        }
        else {
            s_values[threadIdx.x] = 0;
            s_indices[threadIdx.x] = -1; // Invalid index
        }

        __syncthreads();

        int end_point = min(blockDim.x, number_nonzeros_in_row - (blockDim.x * j));
        if (COL < ncols) {
            for (int i = 0; i < end_point; i++)  {
                if (s_indices[i] != -1) {
                    float x_value = X[COL + ncols * s_indices[i]];
                    int activated = (signbit(x_value) * -2) + 1;
                    total_interaction -= s_values[i] * activated;
                } else {
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    if (COL < ncols && !confined) {
        int index = COL + ROW * ncols;

        // based on interactions, update the positions and momenta.
        Y[index] += time_step * xi0 * total_interaction;


        X[index] += time_step * (pressure + 1) * Y[index];
        Y[index] += time_step * (pressure - 1) * X[index];
    }
}

__global__ void compute_max_cut(int *a_pointers, int *a_indices, double *a_values, int *final_spins, double *cut_array, int ncols, int nrows)
{
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    double tmpSum = 0;

    // calculates the energy of the Hamiltonian according to the following formula:
    // J = interaction matrix, x = spin arrangement
    // 0.5 * x * J * t(x) (all matrix multiplication)

    if (ROW < nrows && COL < ncols)
    {
        int start_pointer = a_pointers[ROW];
        int end_pointer = a_pointers[ROW + 1];
        for (int i = start_pointer; i < end_pointer; i++)
        {
            tmpSum += a_values[i] * final_spins[a_indices[i] * ncols + COL];
        }
        cut_array[ROW * ncols + COL] = tmpSum * final_spins[ROW * ncols + COL] * (0.5);
    }
}

__global__ void sum_max_cut(double *cut_array, double *final_cut_array, double sum, int ncols, int nrows) {
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    double tmpSum = 0;

    // calculates the actual maximum cut according to the following formula:
    // H = Hamiltonian energy, S = sum of all weights in MAXCUT graph
    // -0.5 * (H - S)

    if (ROW == 0 && COL < ncols) {
        for (int i = 0; i < nrows; i++) {
            tmpSum += cut_array[i*ncols + COL];
        }
        final_cut_array[COL] = (-0.5) * (tmpSum - sum);
        // final_cut_array[COL] = tmpSum;
    }
}

// Traveling Salesman Problem (TSP)-specific kernels

__global__ void TSP_distances_GEO(double* distances, double* latitudes, double* longitudes, int ncities)
{
    int city1 = blockIdx.x * blockDim.x + threadIdx.x;
    int city2 = blockIdx.y * blockDim.y + threadIdx.y;

    // calculates the distances between all unique pairs of cities according to an adaptation of the Haversine formula
    // the formula used here is from the TSPLIB documentation for "GEO" type distances

    if (city2 < ncities && city1 < city2)
    {
        double R = 6378.388;
        double q1 = cos(longitudes[city1] - longitudes[city2]);
        double q2 = cos(latitudes[city1] - latitudes[city2]);
        double q3 = cos(latitudes[city1] + latitudes[city2]);

        double d = ( R * acos( 0.5 * (( 1.0 + q1 ) * q2 - (1.0 - q1) * q3 )) + 1.0);
        int position = city2 + (city1 * ncities) - ((city1+2)*(city1+1)/ 2);
        distances[position] = (int) d;
    }
}

__global__ void TSP_distances_EUC_2D(double* distances, double* latitudes, double* longitudes, int ncities)
{
    int city1 = blockIdx.x * blockDim.x + threadIdx.x;
    int city2 = blockIdx.y * blockDim.y + threadIdx.y;

    // calculates the distances between all unique pairs of cities according to the euclidean formula
    // the formula used here is from the TSPLIB documentation for "EUC_2D" type distances

    if (city2 < ncities && city1 < city2)
    {
        double xd = latitudes[city1] - latitudes[city2];
        double yd = longitudes[city1] - longitudes[city2];
        double d = pow(xd*xd + yd*yd, 0.5);

        int position = city2 + (city1 * ncities) - ((city1+2)*(city1+1)/ 2);
        distances[position] = (int) d;
    }
}

__global__ void tsp_spin_average(int *spin_array, double* average_array, int ncities, int ncols, int nrows) {
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    // a debugging kernel to determine what the average value for a 
    // particular spin is across the number of agents being run for a TSP graph

    double sum = 0;

    if (ROW < nrows && COL < 1) {
        for (int i = 0; i < ncols; i++) {
            sum += spin_array[ROW * ncols + i];
        }
        average_array[ROW] = sum / ncols;
    }
}

__global__ void update_window_tsp(double *X, int *current_spins, int *final_spins, bool *bifurcated, bool *prev_bifurcated, int* stability, int* valid_list, int convergence_threshold, int* city_visits, int* order_visits, int* step, int ncities, int ncols, int nrows)
{
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    // identical to the update_window kernel above, with small differences that will be pointed out

    bool valid = 1;
    int sum = 0;
    bool equal;
    // bool new_bifurcated;

    if (ROW == 0 && COL < ncols)
    {
        for (int i = 0; i < nrows; i++)
        {
            double val = X[ncols * i + COL];
            int x_sign = (signbit(val) * -2) + 1;
            
            // val / abs(val);
            sum += x_sign * current_spins[ncols * i + COL];

            // the following if statement notes the row and column of every single spin with value "1"

            if (x_sign == 1) {
                int cur_col = i % ncities;
                int cur_row = (i - cur_col) / ncities;
                city_visits[cur_col * ncols + COL] += 1;
                order_visits[cur_row * ncols + COL] += 1;
            }
        }

        // if there is more than one "1" per row or column, for any row or column, then the current spin arrangement is not valid

        for (int i = 0; i < ncities; i++) {
            if (city_visits[i*ncols + COL] != 1 || order_visits[i*ncols + COL] != 1) {
                valid = 0;
                break;
            }
        }

        // reset all values used to 0 for the next iteration of the kernel

        for (int i = 0; i < ncities; i++) {
            city_visits[i*ncols + COL] = 0;
            order_visits[i*ncols + COL] = 0;
        }

        equal = (sum == nrows);

        bool not_equal = !equal;
        bool not_bifurcated = !bifurcated[COL];
        
        stability[COL] += equal && not_bifurcated;
        stability[COL] = not_equal && not_bifurcated ? 0 : stability[COL];
        bifurcated[COL] = stability[COL] == convergence_threshold - 1;

        // new_bifurcated = bifurcated[COL] ^ prev_bifurcated[COL];
        prev_bifurcated[COL] = bifurcated[COL];

        // rather than checking for bifurcation for writing spins to the final spins array, we check for validity, 
        // as this is a stronger and more important metric for TSP graphs

        if (valid) {
            valid_list[COL] = step[0];
            for (int i = 0; i < nrows; i++) {
                double val = X[i*ncols + COL];
                int x_sign = (signbit(val) * -2) + 1;
                final_spins[i * ncols + COL] = x_sign;
            }
        }
    }

    if (ROW < nrows && COL < ncols)
    {
        double val = X[ROW * ncols + COL];
        int x_sign = (signbit(val) * -2) + 1;
        // val / abs(val);
        current_spins[ROW * ncols + COL] = x_sign;
    }
}

__global__ void symplectic_kernel_tsp(int *a_pointers, int *a_indices, double *a_values, double *total_distances, double *X, double *Y, double time_step, int* step, int max_steps, double xi0, int n, double A, double B, double C, int ncities, int ncols, int nrows)
{
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    double interaction1 = 0;

    // we initialize time dependent parameters a(t) and b(t)
    // the formulas for these parameters were optimized through trial and error

    double a_t = n * 1.04 * (double) step[0] / max_steps;
    double b_t = 0.5 + ((double) 0.5 * pow( (double) step[0] / (double) max_steps, 1.9));

    // following statements ensures that the parameters never exceed their maximums

    a_t = a_t < n ? a_t : n;
    b_t = b_t < 1 ? b_t : 1;
// 
    if (ROW < nrows && COL < ncols)
    {
        if (ROW == 0 && COL == 0) step[0] += 1;
        double val = X[ROW * ncols + COL];

        if (abs(val) > 1)
        {
            X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
            Y[ROW * ncols + COL] = 0;
        }
        else
        {
            // note that we use the actual value of the position of the oscillator instead of its sign
            // this is ballistic Simulated Bifurcation (bSB) and outperforms discrete Simulated Bifurcation (dSB)
            // there is also a 1.005 multiplier that improves solution quality


            int start_pointer = a_pointers[ROW];
            int end_pointer = a_pointers[ROW + 1];
            for (int i = start_pointer; i < end_pointer; i++)
            {
                double x_value = X[a_indices[i] * ncols + COL];
                // int activated = (signbit(x_value) * -2) + 1;
                double activated = x_value * 1.005;
                interaction1 -= a_values[i] * activated;
            }

            // there is a city dependent second interaction that is gradually increased by the b(t) parameter,
            // such that it allows the system to adiabatically evolve freely before being constrained in the end

            int city_index = ROW % ncities;
            double interaction2 = - 1 * (A * total_distances[city_index] + (ncities-2) * (B + C)) / 2;
            Y[ROW * ncols + COL] += time_step * (xi0 * (2 * interaction1 + b_t * interaction2));
            X[ROW * ncols + COL] += time_step * 1 * Y[ROW * ncols + COL];
            Y[ROW * ncols + COL] += time_step * -(1 - a_t) * X[ROW * ncols + COL];

        }
    }
}

__global__ void symplectic_kernel_tsp_shared_memory(int *a_pointers, int *a_indices, double *a_values, double *total_distances, double *X, double *Y, double time_step, int* step, int max_steps, double xi0, int n, double A, double B, double C, int ncities, int ncols, int nrows)
{

    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + gridDim.y * blockIdx.z;

    if (ROW >= nrows) return;

    double interaction1 = 0;
    // bool confined = false;

    // we initialize time dependent parameters a(t) and b(t)
    // the formulas for these parameters were optimized through trial and error

    double a_t = n * 1.04 * (double) step[0] / max_steps;
    double b_t = 0.5 + ((double) 0.5 * pow( (double) step[0] / (double) max_steps, 1.9));

    // following statements ensures that the parameters never exceed their maximums

    a_t = a_t < n ? a_t : n;
    b_t = b_t < 1 ? b_t : 1;

    if (ROW == 0 && COL == 0) step[0] += 1;
    if (COL < ncols) {
        double val = X[ROW * ncols + COL];

        if (abs(val) > 1)
        {
            X[ROW * ncols + COL] = (signbit(val) * -2) + 1;
            Y[ROW * ncols + COL] = 0;
            // confined = true;
        }
    }

    __shared__ float s_values[1024];
    __shared__ int s_indices[1024];

    int start_pointer = a_pointers[ROW];
    int end_pointer = a_pointers[ROW + 1];

    int number_nonzeros_in_row = end_pointer - start_pointer;


    for (int j = 0; j < ceil((float)number_nonzeros_in_row / blockDim.x); j++)  {
        
        int idx = threadIdx.x + (blockDim.x * j);
        
        if (idx < number_nonzeros_in_row) {
            s_values[threadIdx.x] = a_values[start_pointer + idx];
            s_indices[threadIdx.x] = a_indices[start_pointer + idx];
        }
        else {
            s_values[threadIdx.x] = 0;
            s_indices[threadIdx.x] = -1; // Invalid index
        }

        __syncthreads();

        // note that we use the actual value of the position of the oscillator instead of its sign
        // this is ballistic Simulated Bifurcation (bSB) and outperforms discrete Simulated Bifurcation (dSB)
        // there is also a 1.005 multiplier that improves solution quality

        int end_point = min(blockDim.x, number_nonzeros_in_row - (blockDim.x * j));
        if (COL < ncols) {
            for (int i = 0; i < end_point; i++)  {
                if (s_indices[i] != -1) {
                    float x_value = X[COL + ncols * s_indices[i]];
                    interaction1 -= s_values[i] * x_value * 1.005;
                } else {
                    break;
                }
            }
        }
        __syncthreads();
    }

    // there is a city dependent second interaction that is gradually increased by the b(t) parameter,
    // such that it allows the system to adiabatically evolve freely before being constrained in the end

    if (COL < ncols) {
        int city_index = ROW % ncities;
        double interaction2 = - 1 * (A * total_distances[city_index] + (ncities-2) * (B + C)) / 2;
        Y[ROW * ncols + COL] += time_step * (xi0 * (2 * interaction1 + b_t * interaction2));
        X[ROW * ncols + COL] += time_step * 1 * Y[ROW * ncols + COL];
        Y[ROW * ncols + COL] += time_step * -(1 - a_t) * X[ROW * ncols + COL];
    }
}


