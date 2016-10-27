#include <iostream>		// for cerr and cout

#include "Mnist_Anomaly_Detection.ipp"
#include "Video_Prediction.ipp"
#include "SequenceRecall.ipp"
#include "SparseFeatures.ipp"
int main()
{
	//TEST

	if (false) {
		SparseFeatures::speedTest(100);
		PredictorLayer::speedTest(100);
	}
	if (true) mnist::mnist_Anomaly_Detection();
	if (false) video::video_Prediction();
	if (false) sequenceRecall();

	std::cout << "Press any key to close" << std::endl;
	//getchar();
	return 0;
}