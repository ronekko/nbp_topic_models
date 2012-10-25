#pragma once
#include "stdafx.h"

/////////////////////////////////////////////////////////////////////////
// NB-LDA in
// Mingyuan Zhou and Lawrence Carin,
// "Negative Binomial Process Count and Mixture Modeling", 2012
/////////////////////////////////////////////////////////////////////////
class NBLDA
{
public:
	NBLDA(const std::vector<std::vector<int>> &corpus,
								const int &V,
								const int &K = 12,
								const int &seed = 0);
	NBLDA(void);
	~NBLDA(void);
	void train(int iter);
	void sample_z(void);
	void sample_p(void);
	void sample_l(void);
	void sample_l_prime(void);
	void sample_gamma_0(void);
	void sample_r(void);
	void sample_theta(void);
	void sample_phi(void);
	
	void set_counts_from_z(void);
	
	std::vector<std::vector<int>> corpus;
	int V;
	int K;
	int M;
	std::vector<int> N; // N_m: number of words in the m-th document.
	boost::mt19937 engine;

	double c;
	double eta;
	double a0;
	double b0;
	double e0;
	double f0;

	double gamma0;
	std::vector<std::vector<int>> z;		// z[m][i]
	std::vector<double> p;					// p[m]
	std::vector<double> p_prime;			// p_prime[m]
	std::vector<std::vector<int>> l;		// l[m][k]
	std::vector<int> l_prime;				// l_prime[m]
	std::vector<double> r;					// r[m]
	std::vector<std::vector<double>> theta;	// theta[m][k]
	std::vector<std::vector<double>> phi;	// phi[k][v]

	std::vector<std::vector<int>> n_km;
	std::vector<std::vector<int>> n_kv;
	std::vector<int> n_k;
};