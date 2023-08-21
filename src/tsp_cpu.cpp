#include "csr.cuh"

TSP_CPU::TSP_CPU(string filename)
{

	read_file(filename, desired_cities);
}

void TSP_CPU::read_file(string filename, int cities_to_read)
{
	std::ifstream fin(filename);
	// Ignore headers and comments:

	while (fin.peek() != 'D')
		fin.ignore(2048, '\n');
	string filler;
	fin >> filler >> ncities;
	std::cout << ncities << "\n";
	while (fin.peek() != 'E')
		fin.ignore(2048, '\n');
	fin >> filler >> dtype;
	std::cout << dtype << "\n";

	if (override_ncities) ncities = cities_to_read;
	double PI = 3.14159265358979323846;

	while (fin.peek() != '1')
		fin.ignore(2048, '\n');
	// Read defining parameters:
	for (int i = 0; i < ncities; i++) {
		double n, lat_coord, long_coord;
		fin >> n >> lat_coord >> long_coord;
		if (dtype == "GEO") {
			int lat_deg = (int) lat_coord, long_deg = (int) long_coord;
			double lat_min = lat_coord - lat_deg, long_min = long_coord - long_deg;
			lat_coord = PI * (lat_deg + 5.0 * lat_min / 3.0) / 180.0;
			long_coord = PI * (long_deg + 5.0 * long_min / 3.0) / 180.0;
		}
		latitudes.push_back(lat_coord);
		longitudes.push_back(long_coord);
	}

	fin.close();
	nonzeros = (ncities*(ncities - 1)) / 2;
}

