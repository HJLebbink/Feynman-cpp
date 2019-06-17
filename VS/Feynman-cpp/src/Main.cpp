#include <iostream>		// for cerr and cout

#include "FixedPoint.ipp"
#include "Mnist_Anomaly_Detection.ipp"
#include "Video_Prediction.ipp"
//#include "SequenceRecall.ipp"
//#include "SparseFeatures.ipp"
//#include "RecallTests.ipp"
//#include "SparseCoderTests.ipp"
//#include "ScalarEncoderTests.ipp"
#include "TextPrediction.ipp"
//#include "SpectrumPrediction.ipp"
#include "LoadPython.ipp"
#include "PlotDebug.ipp"

int main()
{
	//TEST
	if (false) {
		//SparseFeatures::speedTest(1000);
		//PredictorLayer::speedTest(1000);
	}
	if (false) mnist::mnist_Anomaly_Detection();
	if (true) video::video_Prediction();
	if (false) feynman::text_Prediction();
	//if (false) feynman::spectrum_Prediction();
	//if (false) sequenceRecall();
	//if (false) recallTest_AAAX();
	//if (false) sparseCoderTests::SparseCoderTests();
	//if (true) scalarEncoderTests::ScalarEncoderTests();

	std::cout << "Press any key to close" << std::endl;
	std::cin.ignore();
	return 0;
}