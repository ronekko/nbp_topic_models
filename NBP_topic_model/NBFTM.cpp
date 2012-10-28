#include "stdafx.h"
#include "NBFTM.h"

using namespace std;

NBFTM::NBFTM(const vector<vector<int>> &corpus,
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
	
	pi = vector<double>(K);
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

	// ‰B‚ê•Ï”‚Ì‰Šú‰»
	z = vector<vector<int>>(M);
	b = vector<vector<char>>(M);
	for(int m=0; m<M; ++m){
		z[m] = vector<int>(N[m]);
		b[m] = vector<char>(K, 0);
		b[m][0] = 1;
		for(int i=0; i<N[m]; ++i){
			int k = boost::uniform_int<>(0, K-1)(engine);
			z[m][i] = k;
		}
	}

	set_counts_from_z();

	gamma0 = util::gammaRandom(engine, e0, 1.0 / f0);
	for(int k=0; k<K; ++k){
		//p[m] = util::betaRandom(engine, a0, b0);
		pi[k] = 0.5;
		r[k] = util::gammaRandom(engine, gamma0, 1.0 / c);
	}

	//test_CRTRandom();
}

NBFTM::NBFTM(void){}
NBFTM::~NBFTM(void){}


void NBFTM::set_counts_from_z(void)
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

double NBFTM::calc_perplexity(void)
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


void NBFTM::show_parameters(void)
{
	for(int k=0; k<K; ++k){
		cout << k << ":";
		printf("%.3f ", pi[k]);
	}
	cout << endl;
	for(int k=0; k<K; ++k){
		cout << k << ":";
		printf("%.3f ", r[k]);
	}
	cout << endl;

	//for(int m=0; m<M; ++m){
	//	if(m%100 == 0){
	//		boost::copy(b[m], ostream_iterator<unsigned char>(cout, "")); cout <<endl;
	//	}
	//}
}


void NBFTM::train(int iter)
{
	sample_pi();
	sample_b();
	sample_l();
	sample_l_prime();
	sample_gamma_0();
	sample_r();
	sample_theta();
	sample_phi();
	sample_z();
}

void NBFTM::sample_z(void)
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

void NBFTM::sample_pi(void)
{
	for(int k=0; k<K; ++k){
		int b_k = 0;
		for(int m=0; m<M; ++m){
			b_k += b[m][k];
		}
		double a = c / K + b_k;
		double b = c / (1.0 - 1.0/K) + M - b_k;
		pi[k] = util::betaRandom(engine, a, b);
		p_prime[k] = b_k * log(1.0 - 0.5) / (b_k * log(1.0 - 0.5) - c);
	}
}


void NBFTM::sample_b(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			if(n_km[k][m] == 0){
				double tmp = pow(0.5, r[k]);
				double p = pi[k] * tmp / (pi[k] * tmp + 1.0 - pi[k]);
				b[m][k] = boost::bernoulli_distribution<>(p)(engine);
			}
			else{
				b[m][k] = 1;
			}
		}
	}
}


void NBFTM::sample_l(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			l[m][k] = util::CRTRandom(engine, n_km[k][m], b[m][k] * r[k]);
		}
	}
}

void NBFTM::sample_l_prime(void)
{
	for(int k=0; k<K; ++k){
		int l_m_sum = boost::accumulate(l[k], 0);
		l_prime[k] = util::CRTRandom(engine, l_m_sum, gamma0);
	}
}

void NBFTM::sample_gamma_0(void)
{
	double shape = e0 + boost::accumulate(l_prime, 0);
	double scale = 1.0 / (f0 - boost::accumulate(p_prime, 0.0, [](double sum, double x){ return sum + log(1.0 - x);}));
	gamma0 = util::gammaRandom(engine, shape, scale);	
}

void NBFTM::sample_r(void)
{
	for(int k=0; k<K; ++k){
		int b_k = 0;
		for(int m=0; m<M; ++m){
			b_k += b[m][k];
		}
		double shape = gamma0 + boost::accumulate(l[k], 0);
		double scale = 1.0 / (c - b_k * log(1.0 - 0.5));
		r[k] = util::gammaRandom(engine, shape, scale);
	}
}

void NBFTM::sample_theta(void)
{
	for(int m=0; m<M; ++m){
		for(int k=0; k<K; ++k){
			double shape = b[m][k] * r[k] + n_km[k][m];
			double scale = 0.5;
			theta[m][k] = util::gammaRandom(engine, shape, scale);
		}
	}
}

void NBFTM::sample_phi(void)
{
	for(int k=0; k<K; ++k){
		vector<double> eta_post(V, eta);
		for(int v=0; v<V; ++v){
			eta_post[v] += n_kv[k][v];
		}
		phi[k] = util::dirichletRandom(engine, eta_post);
	}
}