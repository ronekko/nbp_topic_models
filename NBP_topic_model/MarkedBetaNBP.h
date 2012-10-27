#pragma once
#include "stdafx.h"

/////////////////////////////////////////////////////////////////////////
// Marked-Beta-negative binomial process in
// Mingyuan Zhou and Lawrence Carin,
// "Negative Binomial Process Count and Mixture Modeling", 2012.
/////////////////////////////////////////////////////////////////////////
class MarkedBetaNBP
{
public:
	MarkedBetaNBP(const std::vector<std::vector<int>> &corpus,
								const int &V,
								const int &K = 200,
								const int &seed = 0);
	MarkedBetaNBP(void);
	~MarkedBetaNBP(void);
	double calc_perplexity(void);
	void show_parameters(void);
	void train(int iter);
	void sample_z(void);
	void sample_p(void);
	void sample_r(void);
	void sample_theta(void);
	void sample_phi(void);
	
	void set_counts_from_z(void);
	
	std::vector<std::vector<int>> corpus;
	int V;
	int K;
	int M;
	std::vector<int> N; // N[m]: number of words in the m-th document.
	int N_total;		// 全文書の全トークン数
	std::vector<std::vector<int>> y;		// y[m][v]: 文書m中の単語vの出現回数
	int iteration;
	boost::mt19937 engine;

	double c;
	double eta;
	double e0;
	double f0;

	double gamma0;
	std::vector<std::vector<int>> z;		// z[m][i]
	std::vector<double> p;					// p[k]
	std::vector<double> r;					// r[k]
	std::vector<std::vector<double>> theta;	// theta[m][k]
	std::vector<std::vector<double>> phi;	// phi[k][v]

	std::vector<std::vector<int>> n_km;
	std::vector<std::vector<int>> n_kv;
};