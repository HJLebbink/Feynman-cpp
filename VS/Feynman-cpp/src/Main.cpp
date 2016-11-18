#include <iostream>		// for cerr and cout

#include "FixedPoint.ipp"
#include "Mnist_Anomaly_Detection.ipp"
#include "Video_Prediction.ipp"
#include "SequenceRecall.ipp"
#include "SparseFeatures.ipp"
#include "RecallTests.ipp"
#include "SparseCOderTests.ipp"


int main()
{
	//TEST
	if (false) fixedPointTest::test2();

	if (false) {
		SparseFeatures::speedTest(1000);
		PredictorLayer::speedTest(1000);
	}
	if (false) mnist::mnist_Anomaly_Detection();
	if (false) video::video_Prediction();
	if (false) sequenceRecall();
	if (false) recallTest_AAAX();
	if (true) sparseCoderTests::SparseCoderTests();


	std::cout << "Press any key to close" << std::endl;
	std::cin.ignore();
	return 0;
}