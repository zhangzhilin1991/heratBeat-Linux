#include "js_utils.h"

int get_len(nc::NdArray<double> input) {
	int len = 0;
	for (auto it = input.begin(); it < input.end(); ++it) {
		len++;
	}
	return len;
}

double get_hamming(double n, double M) {
	return 0.54 - 0.46 * cos(2 * M_PI * n / (M));
}

nc::NdArray<double> hamming(double M) {
	vector<double> tmp;
	for (int i = 0; i < M; i++) {
		tmp.push_back(get_hamming(i, M)); 
	}
	return nc::NdArray<double>(tmp);
}

vector<double> get_angle(vector<complex<double>> raw) {
	vector<double> result(raw.size());
	vector<double>::iterator result_it = result.begin();
	for (vector<complex<double>>::iterator raw_it = raw.begin(); raw_it != raw.end(); raw_it++) {
		*result_it = std::arg(*raw_it);
		result_it++;
	}
	return result;
}

vector<double> get_abs(vector<complex<double>> raw) {
	vector<double> result(raw.size());
	vector<double>::iterator result_it = result.begin();
	for (vector<complex<double>>::iterator raw_it = raw.begin(); raw_it != raw.end(); raw_it++) {
		*result_it = std::abs(*raw_it);
		result_it++;
	}
	return result;
}

void delete_rows(Mat &src) {
	flip(src, src, 0);
	src.pop_back();
	flip(src, src, 0);
}