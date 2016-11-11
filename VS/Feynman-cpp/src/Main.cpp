#include <iostream>		// for cerr and cout

#include "Mnist_Anomaly_Detection.ipp"
#include "Video_Prediction.ipp"
#include "SequenceRecall.ipp"
#include "SparseFeatures.ipp"
#include "RecallTests.ipp"

int main()
{
	//TEST

	if (false) {
		SparseFeatures::speedTest(50);
		//PredictorLayer::speedTest(20);
	}
	if (false) mnist::mnist_Anomaly_Detection();
	if (true) video::video_Prediction();
	if (false) sequenceRecall();
	if (false) recallTest_AAAX();

	std::cout << "Press any key to close" << std::endl;
	std::cin.ignore();
	return 0;
}