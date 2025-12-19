#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <utility>

int number_bacteria;
char** bacteria_name;
long M, M1, M2;
short code[27] = { 0, 2, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 1, 17, 18, 5, 19, 3};
#define encode(ch)		code[ch-'A']
#define LEN				6
#define AA_NUMBER		20
#define	EPSILON			1e-010
#define _POSIX_C_SOURCE 199309L

static inline void set_omp_threads(int n) {
    omp_set_dynamic(0);           
    omp_set_max_active_levels(1); 
    omp_set_num_threads(n); 
}

static int fopen_x(FILE** fp, const char* path, const char* mode) {
	#ifdef _MSC_VER
		return fopen_s(fp, path, mode);
	#else
		*fp = fopen(path, mode);
		return (*fp == NULL) ? errno : 0;
	#endif
}

std::vector<double> seq_results;
std::vector<double> par_results;

static inline bool compare_results(const std::vector<double>& a,
                                   const std::vector<double>& b,
                                   double tol, double* max_abs_out) {
    if (a.size() != b.size()) return false;
    double mx = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        double d = fabs(a[k] - b[k]);
        if (d > mx) mx = d;
        if (d > tol) { if (max_abs_out) *max_abs_out = mx; return false; }
    }
    if (max_abs_out) *max_abs_out = mx;
    return true;
}

static inline size_t pair_index(int i, int j, int N) {
    return (size_t)i * (2 * N - i - 1) / 2 + (j - i - 1);
}

void Init()
{
	M2 = 1;
	for (int i=0; i<LEN-2; i++)
		M2 *= AA_NUMBER;
	M1 = M2 * AA_NUMBER;
	M  = M1 * AA_NUMBER;
}

class Bacteria
{
private:
	long* vector;
	long* second;
	long one_l[AA_NUMBER];
	long indexs;
	long total;
	long total_l;
	long complement;

	void InitVectors()
	{
		vector = new long [M];
		second = new long [M1];
		memset(vector, 0, M * sizeof(long));
		memset(second, 0, M1 * sizeof(long));
		memset(one_l, 0, AA_NUMBER * sizeof(long));
		total = 0;
		total_l = 0;
		complement = 0;
	}

	void init_buffer(char* buffer)
	{
		complement++;
		indexs = 0;
		for (int i=0; i<LEN-1; i++)
		{
			short enc = encode(buffer[i]);
			one_l[enc]++;
			total_l++;
			indexs = indexs * AA_NUMBER + enc;
		}
		second[indexs]++;
	}

	void cont_buffer(char ch)
	{
		short enc = encode(ch);
		one_l[enc]++;
		total_l++;
		long index = indexs * AA_NUMBER + enc;
		vector[index]++;
		total++;
		indexs = (indexs % M2) * AA_NUMBER + enc;
		second[indexs]++;
	}

public:
	long count;
	double* tv;
	long *ti;

	Bacteria(char* filename)
	{
		FILE* bacteria_file;
		int OK = fopen_x(&bacteria_file, filename, "r");

		if (OK != 0)
		{
			fprintf(stderr, "Error: failed to open file %s\n", filename);
			exit(1);
		}

		InitVectors();

		char ch;
		while ((ch = fgetc(bacteria_file)) != EOF)
		{
			if (ch == '>')
			{
				while (fgetc(bacteria_file) != '\n');

				char buffer[LEN-1];
				fread(buffer, sizeof(char), LEN-1, bacteria_file);
				init_buffer(buffer);
			}
			else if (ch != '\n')
				cont_buffer(ch);
		}

		long total_plus_complement = total + complement;
		double total_div_2 = total * 0.5;
		int i_mod_aa_number = 0;
		int i_div_aa_number = 0;
		long i_mod_M1 = 0;
		long i_div_M1 = 0;

		double one_l_div_total[AA_NUMBER];
		for (int i=0; i<AA_NUMBER; i++)
			one_l_div_total[i] = (double)one_l[i] / total_l;

		double* second_div_total = new double[M1];
		for (int i=0; i<M1; i++)
			second_div_total[i] = (double)second[i] / total_plus_complement;

		count = 0;
		double* t = new double[M];

		for(long i=0; i<M; i++)
		{
			double p1 = second_div_total[i_div_aa_number];
			double p2 = one_l_div_total[i_mod_aa_number];
			double p3 = second_div_total[i_mod_M1];
			double p4 = one_l_div_total[i_div_M1];
			double stochastic =  (p1 * p2 + p3 * p4) * total_div_2;

			if (i_mod_aa_number == AA_NUMBER-1)
			{
				i_mod_aa_number = 0;
				i_div_aa_number++;
			}
			else
				i_mod_aa_number++;

			if (i_mod_M1 == M1-1)
			{
				i_mod_M1 = 0;
				i_div_M1++;
			}
			else
				i_mod_M1++;

			if (stochastic > EPSILON)
			{
				t[i] = (vector[i] - stochastic) / stochastic;
				count++;
			}
			else
				t[i] = 0;
		}

		delete[] second_div_total;
		delete[] vector;
		delete[] second;

		tv = new double[count];
		ti = new long[count];

		int pos = 0;
		for (long i=0; i<M; i++)
		{
			if (t[i] != 0)
			{
				tv[pos] = t[i];
				ti[pos] = i;
				pos++;
			}
		}
		delete[] t;

		fclose (bacteria_file);
	}
};

class Bacteria_par
{
private:
	uint32_t* vector;
	uint32_t* second;
	long one_l[AA_NUMBER];
	long indexs;
	long total;
	long total_l;
	long complement;

	void InitVectors()
	{
		vector = new uint32_t[M];
		second = new uint32_t[M1];
		memset(vector, 0, M * sizeof(uint32_t));
		memset(second, 0, M1 * sizeof(uint32_t));	
		memset(one_l, 0, AA_NUMBER * sizeof(long));
		total = 0;
		total_l = 0;
		complement = 0;
	}

	void init_buffer(char* buffer)
	{
		complement++;
		indexs = 0;
		for (int i=0; i<LEN-1; i++)
		{
			short enc = encode(buffer[i]);
			one_l[enc]++;
			total_l++;
			indexs = indexs * AA_NUMBER + enc;
		}
		second[indexs]++;
	}

	void cont_buffer(char ch)
	{
		short enc = encode(ch);
		one_l[enc]++;
		total_l++;
		long index = indexs * AA_NUMBER + enc;
		vector[index]++;
		total++;
		indexs = (indexs % M2) * AA_NUMBER + enc;
		second[indexs]++;
	}

public:
	long count;
	double* tv;
	long *ti;

	Bacteria_par(char* filename)
	{
		FILE* bacteria_file;
		int OK = fopen_x(&bacteria_file, filename, "r");

		if (OK != 0)
		{
			fprintf(stderr, "Error: failed to open file %s\n", filename);
			exit(1);
		}

		InitVectors();

		char ch;
		while ((ch = fgetc(bacteria_file)) != EOF)
		{
			if (ch == '>')
			{
				while (fgetc(bacteria_file) != '\n');

				char buffer[LEN-1];
				fread(buffer, sizeof(char), LEN-1, bacteria_file);
				init_buffer(buffer);
			}
			else if (ch != '\n')
				cont_buffer(ch);
		}

		long total_plus_complement = total + complement;
		double total_div_2 = total * 0.5;

		double one_l_div_total[AA_NUMBER];

		#pragma omp parallel for schedule(static)
		for (int i=0; i<AA_NUMBER; i++)
			one_l_div_total[i] = (double)one_l[i] / total_l;

		double* second_div_total = new double[M1];

		#pragma omp parallel for schedule(static)
		for (int i=0; i<M1; i++)
			second_div_total[i] = (double)second[i] / total_plus_complement;

		const double* sdt = second_div_total;
		const double* o1  = one_l_div_total;
		const uint32_t*   vec = vector;

		int T = omp_get_max_threads();
		std::vector<std::vector<std::pair<long,double>>> tbufs(T);

		#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			auto& buf = tbufs[tid];
			buf.reserve(1<<15);

			#pragma omp for schedule(static, 8192) nowait
			for (long i = 0; i < M; ++i) {
				long i_div_aa_number = i / AA_NUMBER;
				int  i_mod_aa_number = i % AA_NUMBER;
				long i_div_M1        = i / M1;
				long i_mod_M1        = i % M1;

				double p1 = sdt[i_div_aa_number];
				double p2 = o1[i_mod_aa_number];
				double p3 = sdt[i_mod_M1];
				double p4 = o1[i_div_M1];
				double stochastic = (p1 * p2 + p3 * p4) * total_div_2;

				if (stochastic > EPSILON) {
					double v = (vec[i] - stochastic) / stochastic;
					if (v != 0.0) buf.emplace_back(i, v);
				}
			}
		}

		// size and allocate final arrays after parallel sweep
		long total_elems = 0;
		std::vector<long> base(T + 1, 0);
		for (int t = 0; t < T; ++t) total_elems += (long)tbufs[t].size();
		count = total_elems;

		tv = new double[count];
		ti = new long[count];
		
		long offset = 0;
		// copy from thread buffers to final arrays
		#pragma omp parallel for schedule(static)
		for (int t = 0; t < T; ++t) {
			auto& buf = tbufs[t];
			for (size_t k = 0; k < buf.size(); ++k) {
				ti[offset + (long)k] = buf[k].first;
				tv[offset + (long)k] = buf[k].second;
			}
			offset += (long)buf.size();
		}

		for (int t = 0; t < T; ++t) {
			std::vector<std::pair<long,double>>().swap(tbufs[t]);
		}

		delete[] second_div_total;
		delete[] vector;
		delete[] second;

		fclose (bacteria_file);
	}
};

void ReadInputFile(const char* input_name)
{
	FILE* input_file;
	int OK = fopen_x(&input_file, input_name, "r");

	if (OK != 0)
	{
		fprintf(stderr, "Error: failed to open file %s (Hint: check your working directory)\n", input_name);
		exit(1);
	}

	if (fscanf(input_file, "%d", &number_bacteria) != 1)
    {
        fprintf(stderr, "Error: malformed list file (missing count)\n");
        exit(1);
    }
	bacteria_name = new char*[number_bacteria];

	for(long i=0;i<number_bacteria;i++)
	{
		char name[10];
		if (fscanf(input_file, "%9s", name) != 1)
        {
            fprintf(stderr, "Error: malformed list file (bad name at line %ld)\n", i + 2);
            exit(1);
        }
		bacteria_name[i] = new char[20];
		snprintf(bacteria_name[i], 20, "data/%s.faa", name);
	}
	fclose(input_file);
}

double CompareBacteria(Bacteria* b1, Bacteria* b2)
{
	double correlation = 0;
	double vector_len1=0;
	double vector_len2=0;
	long p1 = 0;
	long p2 = 0;
	while (p1 < b1->count && p2 < b2->count)
	{
		long n1 = b1->ti[p1];
		long n2 = b2->ti[p2];
		if (n1 < n2)
		{
			double t1 = b1->tv[p1];
			vector_len1 += (t1 * t1);
			p1++;
		}
		else if (n2 < n1)
		{
			double t2 = b2->tv[p2];
			p2++;
			vector_len2 += (t2 * t2);
		}
		else
		{
			double t1 = b1->tv[p1++];
			double t2 = b2->tv[p2++];
			vector_len1 += (t1 * t1);
			vector_len2 += (t2 * t2);
			correlation += t1 * t2;
		}
	}
	while (p1 < b1->count)
	{
		long n1 = b1->ti[p1];
		double t1 = b1->tv[p1++];
		vector_len1 += (t1 * t1);
	}
	while (p2 < b2->count)
	{
		long n2 = b2->ti[p2];
		double t2 = b2->tv[p2++];
		vector_len2 += (t2 * t2);
	}

	return correlation / (sqrt(vector_len1) * sqrt(vector_len2));
}

double CompareBacteria_par(Bacteria_par* b1, Bacteria_par* b2)
{
	double correlation = 0;
	double vector_len1=0;
	double vector_len2=0;
	long p1 = 0;
	long p2 = 0;
	while (p1 < b1->count && p2 < b2->count)
	{
		long n1 = b1->ti[p1];
		long n2 = b2->ti[p2];
		if (n1 < n2)
		{
			double t1 = b1->tv[p1];
			vector_len1 += (t1 * t1);
			p1++;
		}
		else if (n2 < n1)
		{
			double t2 = b2->tv[p2];
			p2++;
			vector_len2 += (t2 * t2);
		}
		else
		{
			double t1 = b1->tv[p1++];
			double t2 = b2->tv[p2++];
			vector_len1 += (t1 * t1);
			vector_len2 += (t2 * t2);
			correlation += t1 * t2;
		}
	}
	while (p1 < b1->count)
	{
		long n1 = b1->ti[p1];
		double t1 = b1->tv[p1++];
		vector_len1 += (t1 * t1);
	}
	while (p2 < b2->count)
	{
		long n2 = b2->ti[p2];
		double t2 = b2->tv[p2++];
		vector_len2 += (t2 * t2);
	}

	return correlation / (sqrt(vector_len1) * sqrt(vector_len2));
}

void CompareAllBacteria()
{
	seq_results.clear();
	Bacteria** b = new Bacteria*[number_bacteria];

    for(int i=0; i<number_bacteria; i++)
	{
		printf("load %d of %d\n", i+1, number_bacteria);
		b[i] = new Bacteria(bacteria_name[i]);
	}

    for(int i=0; i<number_bacteria-1; i++)
		for(int j=i+1; j<number_bacteria; j++)
		{
			printf("%2d %2d -> ", i, j);
			double correlation = CompareBacteria(b[i], b[j]);
			seq_results.push_back(correlation);
			printf("%.20lf\n", correlation);
		}
	for (int i=0;i<number_bacteria;i++) delete b[i];
	delete[] b;
}

void CompareAllBacteria_par()
{
	par_results.clear();
	Bacteria_par** b = new Bacteria_par*[number_bacteria];

    #pragma omp parallel for schedule(static) 
    for(int i=0; i<number_bacteria; i++)
	{
		printf("load %d of %d\n", i+1, number_bacteria);
		b[i] = new Bacteria_par(bacteria_name[i]);
	}
	const int N = number_bacteria;
	par_results.resize((size_t)N * (N - 1) / 2);

	#pragma omp parallel for schedule(guided,1)
    for(int i=0; i<number_bacteria-1; i++)
		for(int j=i+1; j<number_bacteria; j++)
		{
			printf("%2d %2d -> ", i, j);
			double correlation = CompareBacteria_par(b[i], b[j]);
			par_results[pair_index(i, j, N)] = correlation;
			printf("%.20lf\n", correlation);
		}
	for (int i=0;i<number_bacteria;i++) delete b[i];
	delete[] b;
}

int main(int argc,char * argv[])
{
    set_omp_threads(8);

	struct timespec s_start, s_end, p_start, p_end;


    clock_gettime(CLOCK_MONOTONIC, &s_start);
    Init();
    ReadInputFile("list.txt");
    CompareAllBacteria();
    clock_gettime(CLOCK_MONOTONIC, &s_end);

    double elapsed_seq =
        (s_end.tv_sec  - s_start.tv_sec) +
        (s_end.tv_nsec - s_start.tv_nsec) / 1e9;

    printf("Sequential time elapsed: %.3f seconds\n", elapsed_seq);
	

    clock_gettime(CLOCK_MONOTONIC, &p_start);
	Init();
	ReadInputFile("list.txt");
    CompareAllBacteria_par();
    clock_gettime(CLOCK_MONOTONIC, &p_end);

    double elapsed_par =
        (p_end.tv_sec  - p_start.tv_sec) +
        (p_end.tv_nsec - p_start.tv_nsec) / 1e9;

    printf("Parallel time elapsed:   %.3f seconds\n", elapsed_par);

	
    double max_abs = 0.0;
    bool ok = compare_results(seq_results, par_results, 1e-12, &max_abs);
    printf("Outputs match: %s (max abs = %.3e)\n", ok ? "YES" : "NO", max_abs);
	

    return 0;
}

