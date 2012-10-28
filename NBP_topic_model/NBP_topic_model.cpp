// NBP_topic_model.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "NBLDA.h"
#include "MarkedBetaNBP.h"
#include "NBFTM.h"

using namespace std;


cv::Mat upsample(const cv::Mat &src, const int &scale)
{
	using namespace cv;
	int rows = src.rows * scale;
	int cols = src.cols * scale;
	Mat dst(rows, cols, src.type());

	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			dst.at<float>(i,j) = src.at<float>(i/scale,j/scale);
		}
	}

	return dst;
}
void showTopics(const string &title, const vector<vector<double>> &phi, const int &numColsPerRow = 5)
{
	using namespace cv;
	
	const int K = phi.size();
	const int V = phi[0].size();
	const int COLS = numColsPerRow;
	const int ROWS = ceil(double(K) / double(COLS));
	vector<Mat> phiImages;
	Mat result(ROWS*60, COLS*60, CV_32FC1);
	
	for(int k=0; k<K; ++k){
		Mat phiImage(5, 5, CV_32FC1);
		for(int i=0; i<5; ++i){
			for(int j=0; j<5; ++j){
				phiImage.at<float>(i, j) = static_cast<float>(phi[k][i*5+j]);
			}
		}
		phiImages.push_back(upsample(phiImage, 10.0) * 5.0);
	}

	randu(result, Scalar(0.0), Scalar(1.0));

	for(int k=0; k<K; ++k){
		int row = k / COLS;
		int col = k % COLS;
		Mat roi = result(Rect(col*60+5, row*60+5, 50, 50));
		phiImages[k].copyTo(roi);
	}
	imshow(title, result);
	waitKey(1);
}

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
	const int M = 1000;
	const int N_mean = 200;
	const int V = 25;
	int K_max = 20;
	const double ALPHA = 1.0;
	boost::mt19937 engine;
	vector<vector<double>> theta(M);
	
	// 人工トピックたちの生成
	vector<vector<double>> topics = createTopics();
	const int K = topics.size();
	vector<boost::random::discrete_distribution<>> word_distributions(K);
	for(int k=0; k<K; ++k){
		word_distributions[k] = boost::random::discrete_distribution<>(topics[k]);
	}

	vector<vector<int>> corpus(M);
	vector<double> alpha(K, ALPHA / K);
	for(int m=0; m<M; ++m){
		int N_m = boost::poisson_distribution<>(N_mean)(engine);
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


	//NBLDA learner(corpus, V, K, 11);
	//MarkedBetaNBP learner(corpus, V, K_max, 1111);
	NBFTM learner(corpus, V, K_max, 1111);

	for(int i=0; i<10000; ++i){
		cout << i;
		learner.train(1);
		showTopics("topics", learner.phi,10);
		cout << ": " << learner.calc_perplexity() << endl;
		learner.show_parameters();
	}



	return 0;
}