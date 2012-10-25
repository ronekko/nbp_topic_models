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

	// ‰B‚ê•Ï”‚Ì‰Šú‰»
	z = vector<vector<int>>(M);
	for(int m=0; m<M; ++m){
		z[m] = vector<int>(N[m]);
		for(int i=0; i<N[m]; ++i){
			int k = boost::uniform_int<>(0, K-1)(engine);
			z[m][i] = k;
		}
	}

	set_counts_from_z();
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