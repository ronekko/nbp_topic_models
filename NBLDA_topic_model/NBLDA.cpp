#include "stdafx.h"
#include "NBLDA.h"

using namespace std;
double gammaRandom(boost::mt19937 &engine, const double &shape, const double &scale)
{
	boost::math::gamma_distribution<> dist(shape, scale);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}

double betaRandom(boost::mt19937 &engine, const double &alpha, const double &beta)
{
	boost::math::beta_distribution<> dist(alpha, beta);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}
// 多項分布からのサンプリング、ただしパラメータは正規化されていない（\sum p_iが1とは限らない）
int multinomialByUnnormalizedParameters(boost::mt19937 &engine, const vector<double> &p)
{
	const int K=p.size();
	vector<double> CDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		CDF[k] = z + p[k];
		z = CDF[k];
	}

	double u = boost::uniform_01<>()(engine) * CDF.back();
	for(int k=0; k<K; ++k){
		if(u < CDF[k]){
			return k;
		}
	}
	cout <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	return K-1;
}

double logsumexp (double x, double y, bool flg)
{
	if (flg) return y; // init mode
	if (x == y) return x + 0.69314718055; // log(2)
	double vmin = std::min (x, y);
	double vmax = std::max (x, y);
	if (vmax > vmin + 50) {
		return vmax;
	} else {
		return vmax + std::log (std::exp (vmin - vmax) + 1.0);
	}
}
// 多項分布からのサンプリング、ただしパラメータはlog(p_1), ... , log(p_K)で与えられ、正規化されていない（\sum p_iが1とは限らない）
int multinomialByUnnormalizedLogParameters(boost::mt19937 &engine, const vector<double> &lnp)
{
	const int K=lnp.size();
	vector<double> logCDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		z = logsumexp(z, lnp[k], (k==0));
		logCDF[k] = z;
	}

	double u = log(boost::uniform_01<>()(engine)) + logCDF.back();
	for(int k=0; k<K; ++k){
		if(u < logCDF[k]){
			return k;
		}
	}

	return K-1;
}
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
		for(int l=1; l<=n; ++l){
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
	sample_z();
	sample_p();
	sample_p_prime();
	sample_l();
	sample_l_prime();
	sample_gamma_0();
	sample_r();
	sample_theta();
	sample_phi();
}

void NBLDA::sample_z(void)
{
}

void NBLDA::sample_p(void)
{
}

void NBLDA::sample_p_prime(void)
{
}

void NBLDA::sample_l(void)
{
}

void NBLDA::sample_l_prime(void)
{
}

void NBLDA::sample_gamma_0(void)
{
}

void NBLDA::sample_r(void)
{
}

void NBLDA::sample_theta(void)
{
}

void NBLDA::sample_phi(void)
{
}