#include "stdafx.h"
#include "NBLDA.h"

using namespace std;

NBLDA::NBLDA(const vector<vector<int>> &corpus,
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
	c = 1.0;
	eta = 0.05;
	a0 = b0 = e0 = f0 = 0.01;
	
	p = vector<double>(M);
	p_prime = vector<double>(M);
	l = vector<vector<int>>(M, vector<int>(K, 0));
	l_prime = vector<int>(M);
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

	gamma0 = util::gammaRandom(engine, e0, 1.0 / f0);
	for(int m=0; m<M; ++m){
		//p[m] = util::betaRandom(engine, a0, b0);
		p[m] = 0.5;
		p_prime[m] = K * log(1.0 - p[m]) / (K * log(1.0 - p[m]) - c);
		r[m] = util::gammaRandom(engine, gamma0, 1.0 / c);
	}

	//test_CRTRandom();
}

NBLDA::NBLDA(void){}
NBLDA::~NBLDA(void){}


void NBLDA::set_counts_from_z(void)
{
	n_km = vector<vector<int>>(K, vector<int>(M, 0));
	n_kv = vector<vector<int>>(K, vector<int>(V, 0));
	n_k = vector<int>(K, 0);

	for(int m=0; m<M; ++m){
		for(int i=0; i<N[m]; ++i){
			int k = z[m][i];
			int v = corpus[m][i];
			n_km[k][m]++;
			n_kv[k][v]++;
			n_k[k]++;
		}
	}
}

double NBLDA::calc_perplexity(void)
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


void NBLDA::show_parameters(void)
{
}


void NBLDA::train(int iter)
{
	sample_p();
	sample_l();
	sample_l_prime();
	sample_gamma_0();
	sample_r();
	sample_theta();
	sample_phi();
	sample_z();
}

void NBLDA::sample_z(void)
{
	for(int m=0; m<M; ++m){
		for(int i=0; i<N[m]; ++i){
			int v = corpus[m][i];
			int k_old = z[m][i];

			n_km[k_old][m]--;
			n_kv[k_old][v]--;
			n_k[k_old]--;

			vector<double> mult_params(K);
			for(int k=0; k<K; ++k){
				mult_params[k] = phi[k][v] * theta[m][k];
			}
			int k_new = util::multinomialByUnnormalizedParameters(engine, mult_params);

			z[m][i] = k_new;
			n_km[k_new][m]++;
			n_kv[k_new][v]++;
			n_k[k_new]++;
		}
	}
}

void NBLDA::sample_p(void)
{
	for(int m=0; m<M; ++m){
		double a = a0 + N[m];
		double b = b0 + K + r[m];
		p[m] = util::betaRandom(engine, a, b);
		p_prime[m] = K * log(1.0 - p[m]) / (K * log(1.0 - p[m]) - c);
	}
}

void NBLDA::sample_l(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			l[m][k] = util::CRTRandom(engine, n_km[k][m], r[m]);
		}
	}
}

void NBLDA::sample_l_prime(void)
{
	for(int m=0; m<M; ++m){
		int l_m_sum = boost::accumulate(l[m], 0);
		l_prime[m] = util::CRTRandom(engine, l_m_sum, gamma0);
	}
}

void NBLDA::sample_gamma_0(void)
{
	double shape = e0 + boost::accumulate(l_prime, 0);
	double scale = 1.0 / (f0 - boost::accumulate(p_prime, 0.0, [](double sum, double x){ return sum + log(1.0 - x);}));
	gamma0 = util::gammaRandom(engine, shape, scale);	
}

void NBLDA::sample_r(void)
{
	for(int m=0; m<M; ++m){
		double shape = gamma0 + boost::accumulate(l[m], 0);
		double scale = 1.0 / (c - K * log(1.0 - p[m]));
		r[m] = util::gammaRandom(engine, shape, scale);
	}
}

void NBLDA::sample_theta(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			double shape = r[m] + n_km[k][m];
			double scale = p[m];
			theta[m][k] = util::gammaRandom(engine, shape, scale);
		}
	}
}

void NBLDA::sample_phi(void)
{
	for(int k=0; k<K; ++k){
		vector<double> eta_post(V, eta);
		for(int v=0; v<V; ++v){
			eta_post[v] += n_kv[k][v];
		}
		phi[k] = util::dirichletRandom(engine, eta_post);
	}
}