#pragma once

#include "stdafx.h"

namespace util
{

using namespace std;

inline std::vector<double> dirichletRandom(boost::mt19937 &engine, const std::vector<double> &alpha)
{	
	const int K = alpha.size();
	std::vector<double> y(K);
	double sumY = 0.0;

	for(int k=0; k<K; ++k){
		boost::math::gamma_distribution<> dist(alpha[k], 1.0);
		y[k] = boost::math::quantile(dist, boost::uniform_01<>()(engine));
		//y[k] = boost::gamma_distribution<>(alpha[k], 1.0)(engine);	// shapeパラメータが大きいと落ちる
		sumY += y[k];
	}

	for(int k=0; k<K; ++k){
		y[k] /= sumY;
	}

	return y;
}

inline double gammaRandom(boost::mt19937 &engine, const double &shape, const double &scale)
{
	boost::math::gamma_distribution<> dist(shape, scale);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}

inline double betaRandom(boost::mt19937 &engine, const double &alpha, const double &beta)
{
	boost::math::beta_distribution<> dist(alpha, beta);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}
// 多項分布からのサンプリング、ただしパラメータは正規化されていない（\sum p_iが1とは限らない）
inline int multinomialByUnnormalizedParameters(boost::mt19937 &engine, const vector<double> &p)
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

inline double logsumexp (double x, double y, bool flg)
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
inline int multinomialByUnnormalizedLogParameters(boost::mt19937 &engine, const vector<double> &lnp)
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


};