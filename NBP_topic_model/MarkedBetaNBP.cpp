#include "stdafx.h"
#include "MarkedBetaNBP.h"
#include "utility.hpp"

using namespace std;




MarkedBetaNBP::MarkedBetaNBP(const vector<vector<int>> &corpus,
							const int &V,
							const int &K,
							const int &seed) :
	corpus(corpus),
	V(V),
	K(K),
	M(corpus.size()),
	N(([](const vector<std::vector<int>> &corpus)->vector<int>{
		vector<int> counts(corpus.size());
		for(int i=0; i<counts.size(); ++i){
			counts[i] = corpus[i].size();
		}
		return counts;
	})(corpus))
{
	engine.seed(seed);
	iteration = -1;
	c = 1.0;
	eta = 0.05;
	e0 = f0 = 0.01;
	
	p = vector<double>(M);
	r = vector<double>(M);
	theta = vector<vector<double>>(M, vector<double>(K));
	phi = vector<vector<double>>(K, vector<double>(V));

	N_total = boost::accumulate(N, 0);
	y = vector<vector<int>>(M, vector<int>(V, 0));
	for(int m=0; m<M; ++m){
		for(int i=0; i<N[m]; ++i){
			int v = corpus[m][i];
			y[m][v]++;
		}
	}

	// 隠れ変数の初期化
	z = vector<vector<int>>(M);
	for(int m=0; m<M; ++m){
		z[m] = vector<int>(N[m]);
		for(int i=0; i<N[m]; ++i){
			int k = boost::uniform_int<>(0, K-1)(engine);
			z[m][i] = k;
		}
	}

	set_counts_from_z();

	for(int k=0; k<K; ++k){
		//p[k] = util::betaRandom(engine, c / K, c * (1.0 - 1.0 / K)); // 論文の式81は間違い
		p[k] = 0.5;
		r[k] = util::gammaRandom(engine, e0, 1.0 / f0);
	}

	//test_CRTRandom();
}

MarkedBetaNBP::MarkedBetaNBP(void){}
MarkedBetaNBP::~MarkedBetaNBP(void){}


void MarkedBetaNBP::set_counts_from_z(void)
{
	n_km = vector<vector<int>>(K, vector<int>(M, 0));
	n_kv = vector<vector<int>>(K, vector<int>(V, 0));

	for(int m=0; m<M; ++m){
		for(int i=0; i<N[m]; ++i){
			int k = z[m][i];
			int v = corpus[m][i];
			n_km[k][m]++;
			n_kv[k][v]++;
		}
	}
}

double MarkedBetaNBP::calc_perplexity(void)
{
	double total = 0.0;
	double sum_denominator = 0.0;
	for(int m=0; m<M; ++m){
		for(int v=0; v<V; ++v){
			int y_mv = y[m][v];			
			double sum_numerator = 0.0;
			for(int k=0; k<K; ++k){
				sum_numerator += theta[m][k] * phi[k][v];
				sum_denominator += theta[m][k] * phi[k][v];
			}
			if(y_mv != 0){
				total += y_mv * log(sum_numerator);
			}
		}
	}

	total -= N_total * log(sum_denominator);

	return -total / static_cast<double>(N_total);
	//return exp(-total / static_cast<double>(N_total));
}

void MarkedBetaNBP::show_parameters(void)
{
	for(int k=0; k<K; ++k){
		cout << k << ":";
		printf("%.3f ", p[k]);
	}
	cout << endl;
	for(int k=0; k<K; ++k){
		cout << k << ":";
		printf("%.3f ", r[k]);
	}
	cout << endl;
}


void MarkedBetaNBP::train(int iter)
{
	iteration++;
	sample_p();		cout << " p";
	sample_r();		cout << " r";
	sample_theta();	cout << " theta";
	sample_phi();	cout << " phi";
	sample_z();		cout << " z ";
}

void MarkedBetaNBP::sample_z(void)
{
	for(int m=0; m<M; ++m){
		for(int i=0; i<N[m]; ++i){
			int v = corpus[m][i];
			int k_old = z[m][i];

			n_km[k_old][m]--;
			n_kv[k_old][v]--;

			vector<double> mult_params(K);
			for(int k=0; k<K; ++k){
				mult_params[k] = phi[k][v] * theta[m][k];
			}
			int k_new = util::multinomialByUnnormalizedParameters(engine, mult_params);

			z[m][i] = k_new;
			n_km[k_new][m]++;
			n_kv[k_new][v]++;
		}
	}
}

void MarkedBetaNBP::sample_p(void)
{
	for(int k=0; k<K; ++k){
		double a = (c / K) + static_cast<double>(boost::accumulate(n_km[k], 0));
		double b = c * (1.0 - 1.0 / K) + M * r[k];
		p[k] = util::betaRandom(engine, a, b);
	}
}


void MarkedBetaNBP::sample_r(void)
{
	for(int k=0; k<K; ++k){
		int l_k_sum = 0;
		for(int m=0; m<M; ++m){
			l_k_sum += util::CRTRandom(engine, n_km[k][m], r[k]);
		}
		double shape = e0 + l_k_sum;
		double scale = 1.0 / (f0 - M * log(1.0 - p[k]));
		r[k] = util::gammaRandom(engine, shape, scale);
	}
}

void MarkedBetaNBP::sample_theta(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			double shape = r[k] + n_km[k][m]; // 消えそうなトピックの場合ここが0になることがある
			double scale = p[k];
			theta[m][k] = util::gammaRandom(engine, shape, scale); 
		}
	}
}

void MarkedBetaNBP::sample_phi(void)
{
	for(int k=0; k<K; ++k){
		vector<double> eta_post(V, eta);
		for(int v=0; v<V; ++v){
			eta_post[v] += n_kv[k][v];
		}
		phi[k] = util::dirichletRandom(engine, eta_post);
	}
}