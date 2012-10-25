// NBLDA_topic_model.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "NBLDA.h"

using namespace std;


vector<vector<double>> createTopics(void)
{
	// 0.18 * 5 + 0.005 * 20
	vector<vector<double>> topics;
	const int V = 25;

	for(int i=0; i<5; ++i){
		vector<double> phi(V, 0.005);
		for(int j=0; j<5; ++j){
			phi[i*5+j] = 0.18;
		}
		topics.push_back(phi);
	}

	for(int i=0; i<5; ++i){
		vector<double> phi(V, 0.005);
		for(int j=0; j<5; ++j){
			phi[j*5+i] = 0.18;
		}
		topics.push_back(phi);
	}
	
	{
		vector<double> phi(V, 0.005);
		phi[0] = phi[6] = phi[12] = phi[18] = phi[24] = 0.18;
		topics.push_back(phi);
	}
	{
		vector<double> phi(V, 0.005);
		phi[4] = phi[8] = phi[12] = phi[16] = phi[20] = 0.18;
		topics.push_back(phi);
	}
	//{
	//	vector<double> phi(V, 0.0);
	//	phi[7] = phi[11] = phi[12] = phi[13] = phi[17] = 0.20;
	//	topics.push_back(phi);
	//}
	//{
	//	vector<double> phi(V, 0.0);
	//	phi[6] = phi[8] = phi[12] = phi[16] = phi[18] = 0.20;
	//	topics.push_back(phi);
	//}

	return topics;
}


int _tmain(int argc, _TCHAR* argv[]) 
{
	using namespace std;
	const int M = 200;
	const int V = 25;
	const int K = 12;
	const double ALPHA = 3.0;
	vector<double> alpha(K, ALPHA / K);
	boost::mt19937 engine;
	vector<vector<double>> theta(M);
	
	// 人工トピックたちの生成
	vector<vector<double>> topics = createTopics();
	vector<boost::random::discrete_distribution<>> word_distributions(K);
	for(int k=0; k<K; ++k){
		word_distributions[k] = boost::random::discrete_distribution<>(topics[k]);
	}

	vector<vector<int>> corpus(M);
	for(int m=0; m<M; ++m){
		int N_m = 100;
		corpus[m].resize(N_m);

		vector<double> theta_m = util::dirichletRandom(engine, alpha);
		theta[m] = theta_m;

		boost::random::discrete_distribution<> discrete(theta_m);
		for(int i=0; i<N_m; ++i){
			int k = discrete(engine);
			int v = word_distributions[k](engine);
			corpus[m][i] = v;
		}
	}


	NBLDA nblda(corpus, V, K);

	for(int i=0; i<100; ++i){
		cout << i << endl;
		nblda.train(1);
	}



	return 0;
}