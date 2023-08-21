#include "csr.cuh"

CSR_GPU::CSR_GPU(CSR& csr_cpu) {
    size_t row_index_bytes = csr_cpu.csr_row_indices.size()*sizeof(int);
    size_t column_bytes = csr_cpu.csr_columns.size()*sizeof(int);
    size_t data_bytes = csr_cpu.csr_data.size()*sizeof(double);

    cudaMalloc(&d_row_indices, row_index_bytes);
    cudaMalloc(&d_columns, column_bytes);
    cudaMalloc(&d_data, data_bytes);

    cudaMemcpy(d_row_indices, csr_cpu.csr_row_indices.data(), row_index_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, csr_cpu.csr_columns.data(), column_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, csr_cpu.csr_data.data(), data_bytes, cudaMemcpyHostToDevice);
};