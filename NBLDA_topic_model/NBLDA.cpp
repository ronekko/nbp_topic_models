#include "stdafx.h"
#include "NBLDA.h"

using namespace std;
// ディリクレ分布から乱数を生成する http://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution

// Chinese restaurant table distributionからの乱数生成
int CRTRandom(boost::mt19937 &engine, const int &n, const double &alpha)
{
	if(n < 1){
		return 0;
	}

	int l=1;
	for(int i=1; i<n; ++i){
		double p = alpha / (i + alpha);
		l += boost::bernoulli_distribution<>(p)(engine);
	}
	return l;
}
void test_CRTRandom(void)
{
	boost::mt19937 engine(0);
	double alpha = 4.0;
	int r = 100000000;
	{
		//int n = 5; double pmf[] = {0, 24, 50, 35, 10, 1};
		//int n = 6; double pmf[] = {0, 120, 274, 225, 85, 15, 1};
		//int n = 7; double pmf[] = {0, 720, 1764, 1624, 735, 175, 21, 1};
		//int n = 8; double pmf[] = {0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1};
		int n = 9; double pmf[] = {0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1};
		for(int l=1; l<=n; ++l){	pmf[l] *= pow(alpha, l); }
		double total = boost::accumulate(pmf, 0.0);
		for(int l=1; l<=n; ++l){	pmf[l] /= total; }

		vector<int> result(n+1, 0);
		for(int i=0; i<r; ++i){
			int l = CRTRandom(engine, n, alpha);
			result[l]++;
		}

		vector<double> prop(n+1, 0.0);
		for(int l=1; l<=n; ++l){
			prop[l] = result[l] / static_cast<double>(r);
		}
		
		cout << "empirical\tpmf[l]\t\tdiff" << endl;
		for(int l=0; l<=n; ++l){
			cout << prop[l] << "    \t" << pmf[l] << "    \t" << prop[l] - pmf[l] << endl;
		}
	}
}


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

void NBLDA::train(int iter)
{
	sample_p();
	sample_l();
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
			vector<double> mult_params(K);
			for(int k=0; k<K; ++k){
				mult_params[k] = phi[k][v] * theta[m][k];
			}
			z[m][i] = util::multinomialByUnnormalizedParameters(engine, mult_params);
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
			l[m][k] = CRTRandom(engine, n_km[k][m], r[m]);
		}
	}
}

void NBLDA::sample_l_prime(void)
{
	for(int m=0; m<M; ++m){
		int l_m_sum = boost::accumulate(l[m], 0);
		l_prime[m] = CRTRandom(engine, l_m_sum, gamma0);
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