// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>		// for cerr and cout
#include <ctime>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>

#include "Plot.ipp"
#include "PlotDebug.ipp"

#include "Helpers.ipp"
#include "FeatureHierarchy.ipp"
#include "Predictor.ipp"
#include "ScalarEncoder.ipp"

using namespace feynman;

namespace scalarEncoderTests {

	void ScalarEncoderTests()
	{
		std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

		// --------------------------- Create the Sparse Coder ---------------------------
		// Bottom input width and height (= Mnist data size)
		int bottomWidth = 28;
		int bottomHeight = 28;

		// Predictor hierarchy input width and height (= sparse coder size)
		int hInWidth = 32;
		int hInHeight = 32;

		const int numInputs = 10;
		const int numOutputs = 20;
		const float2 initWeightRange = { 0.0f, 0.001f };
		const int seed = 1234567;

		ScalarEncoder scalarEncoder;
		scalarEncoder.createRandom(numInputs, numOutputs, initWeightRange.x, initWeightRange.y, seed);


		const float activeRatio = 0.02;
		const float alpha = 0.02f;	// learning rate for weights (often 0 for this encoder).
		const float beta = 0.02f;	// learning rate for biases (often 0 for this encoder).

		std::vector<float> inputs(numInputs);

		std::uniform_int_distribution<int> intDist(0, 999);
		for (size_t i = 0; i < inputs.size(); ++i) {
			const float r = static_cast<float>(intDist(generator)) / 1000;
			std::printf("INFO: ScalarEncoderTests: inputs[%llu]=%f\n", i, r);
			inputs[i] = r;
		}

		for (int i = 0; i < 1000; ++i) {
			scalarEncoder.encode(inputs, activeRatio, alpha, beta);
		}

		const std::vector<float> sdr = scalarEncoder.getEncoderOutputs();

		for (size_t i = 0; i < sdr.size(); ++i) {
			std::printf("INFO sdr[%llu]=%f\n", i, sdr[i]);
		}

		scalarEncoder.decode(sdr);
		const std::vector<float> outputs = scalarEncoder.getDecoderOutputs();

		for (size_t i = 0; i < outputs.size(); ++i) {
			std::printf("INFO outputs[%llu]=%f\n", i, outputs[i]);
		}
	}
}