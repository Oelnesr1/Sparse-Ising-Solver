#include "csr.cuh"

CSR::CSR(string filename)
{

	// std::cout << "filename = " << filename << std::endl;

	from_file(filename);
}

CSR::CSR(vector<double> matrix, int dim) {
	from_matrix(matrix, dim);
}

void CSR::from_file(string filename)
{
	int M, N, L;
	vector<int> rows, cols;
	vector<double> vals;
	std::ifstream fin(filename);
	// Ignore headers and comments:

	bool symmetric = true;
	bool read_first_line = false;
	bool pattern = false;
	bool no_diagonal = true;

	while (!read_first_line)
	{
		string word;
		fin >> word;
		if (word == "symmetric")
		{
			symmetric = true;
			read_first_line = true;
		}
		else if (word == "pattern")
		{
			pattern = true;
		}
		else if (word == "integer")
		{
			pattern = false;
		}
		else if (word == "general")
		{
			symmetric = false;
			read_first_line = true;
		}
	}

	fin.ignore(2048, '\n');

	while (fin.peek() == '%')
		fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;
	size = M;
	if (pattern)
	{
		for (int l = 0; l < L; l++)
		{
			int row, col;
			fin >> row >> col;
			if (!no_diagonal || (no_diagonal && row != col)) {
				rows.push_back(row - 1);
				cols.push_back(col - 1);
				vals.push_back(1);	
				sum += 1;
			}
		}
	}
	else
	{
		for (int l = 0; l < L; l++)
		{
			int row, col;
			double d;
			fin >> row >> col >> d;
			if (!no_diagonal || (no_diagonal && row != col)) {
				rows.push_back(row - 1);
				cols.push_back(col - 1);
				vals.push_back(d);
				sum += d;
			}
		}
	}

	fin.close();

	if (symmetric)
	{
		int val_size = vals.size();

		for (int i = 0; i < val_size; i++)
		{
			if (cols[i] != rows[i])
			{
				rows.push_back(cols[i]);
				cols.push_back(rows[i]);
				vals.push_back(vals[i]);
			}
		}
	}

	construct_csr(rows, cols, vals);

	// std::cout << "done reading!\n";
}

vector<double> CSR::get_row(int row)
{
	auto start_index = csr_row_indices[row];
	auto end_index = csr_row_indices[row + 1];

	auto start_iterator = csr_data.begin() + start_index;
	auto end_iterator = csr_data.begin() + end_index;

	vector<double> result(end_index - start_index);
	copy(start_iterator, end_iterator, result.begin());
	return result;
}

vector<int> CSR::get_col_of_row(int row)
{
	auto start_index = csr_row_indices[row];
	auto end_index = csr_row_indices[row + 1];

	auto start_iterator = csr_columns.begin() + start_index;
	auto end_iterator = csr_columns.begin() + end_index;

	vector<int> result(end_index - start_index);
	copy(start_iterator, end_iterator, result.begin());
	return result;
}

void CSR::from_matrix(vector<double> matrix, int dim) {

	vector<int> rows, cols;
	vector<double> vals;

	for (int row = 0; row < dim; row++) {
		for (int col = 0; col < dim; col++) {
			double val = matrix.at(row*dim + col);
			if (val != 0) {
				rows.push_back(row);
				cols.push_back(col);
				vals.push_back(val);
			}
		}
	}
	construct_csr(rows, cols, vals);
	size = dim;

}

void CSR::construct_csr(vector<int> rows, vector<int> cols, vector<double> vals) {
	vector<std::tuple<int, int, double>> tuple_vector;

	for (int i = 0; i < vals.size(); i++)
	{
		auto matrix_tuple = std::make_tuple(rows[i], cols[i], vals[i]);
		tuple_vector.push_back(matrix_tuple);
	}

	std::sort(tuple_vector.begin(), tuple_vector.end());

	int prev_row = 0;
	csr_row_indices.push_back(0);

	for (int i = 0; i < vals.size(); i++)
	{
		int cur_row = std::get<0>(tuple_vector[i]);
		if (cur_row == prev_row)
		{
			csr_columns.push_back(std::get<1>(tuple_vector[i]));
			csr_data.push_back(std::get<2>(tuple_vector[i]));
		}
		else
		{
			csr_row_indices.push_back(csr_columns.size());
			csr_columns.push_back(std::get<1>(tuple_vector[i]));
			csr_data.push_back(std::get<2>(tuple_vector[i]));
			prev_row = cur_row;
		}
	}

	csr_row_indices.push_back(csr_columns.size());

	nonzeros = csr_columns.size();

}