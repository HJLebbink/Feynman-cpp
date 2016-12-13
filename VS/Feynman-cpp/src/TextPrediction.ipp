#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include "Helpers.ipp"
#include "Architect.ipp"
#include "Hierarchy.ipp"
#include "SparseFeaturesSTDP.ipp"
#include "TextConverter.ipp"

using namespace feynman;
using namespace cv;

namespace feynman {

	const std::string inputFileName = "alice-in-wonderland.txt";

	void text_Prediction() {

		std::ifstream inFile;
		inFile.open(inputFileName);//open the input file

		std::stringstream strStream;
		strStream << inFile.rdbuf();//read the file
		std::string example = strStream.str();//str holds the content of the file


		// Target file name
		// --------------------------- Create the Hierarchy ---------------------------
		feynman::Architect arch;
		arch.initialize(1234);

		const int windowWidth = 20;
		const int nNeuronsChar = 200;
		const float sparsity = 0.03;
		const int textJump = 4;

		const int2 inputLayerSize = int2{ windowWidth, nNeuronsChar };

		arch.addInputLayer(inputLayerSize)
			.setValue("in_p_alpha", 0.02f)
			.setValue("in_p_radius", 8);

		{
			for (int l = 0; l < 1; l++)
				arch.addHigherLayer(int2{ 100, 100 }, feynman::_old)
				.setValue("inhibitionRadius", 5)
				.setValue("activeRatio", 0.01f)
				.setValue("biasAlpha", 0.01f)
				.setValue("initWeightRange", float2{ 0.0, 1.0 })
				.setValue("initBiasRange", float2{ -0.01, 0.01 })

				.setValue("ff_radius", 20)
				.setValue("ff_weightAlpha", 0.25f) // used in SparseFeaturesChunk:sfcLearnWeights 

				.setValue("r_radius", 20)
				.setValue("r_weightAlpha", 0.25f)

				.setValue("hl_poolSteps", 1) // used in FeatureHierarchy:fhPool
				.setValue("p_alpha", 0.08f)
				.setValue("p_beta", 0.16f)
				.setValue("p_radius", 8);
		}

		{
			for (int l = 0; l < 0; l++)
				arch.addHigherLayer(int2{ 64, 64 }, feynman::_chunk)
				.setValue("sfc_chunkSize", int2{ 6, 6 })
				.setValue("sfc_initWeightRange", float2{ -1.0, 1.0 })
				.setValue("sfc_numSamples", 2)
				.setValue("sfc_biasAlpha", 0.1f)
				.setValue("sfc_gamma", 0.92f)	// used in SFChunk:sfsInhibit

				.setValue("sfc_ff_radius", 16)
				.setValue("sfc_ff_weightAlpha", 0.3f) // used in SparseFeaturesChunk:sfcLearnWeights 
				.setValue("sfc_ff_lambda", 0)	// input decay
				.setValue("sfc_r_radius", 16)
				.setValue("sfc_r_weightAlpha", 0.3f)
				.setValue("sfc_r_lambda", 0)	// input decay

				.setValue("hl_poolSteps", 2) // used in FeatureHierarchy:fhPool
				.setValue("p_alpha", 0.04f)
				.setValue("p_beta", 0.08f)
				.setValue("p_radius", 8);
		}

		{
			// 4 layers using stdp encoders
			for (int l = 0; l < 0; l++)
				arch.addHigherLayer({ 64, 64 }, feynman::_stdp)
				.setValue("sfs_inhibitionRadius", 6)
				.setValue("sfs_initWeightRange", float2{ -0.01, 0.01 })
				.setValue("sfs_biasAlpha", 0.01f)

				.setValue("sfs_activeRatio", 0.02f)
				.setValue("sfs_gamma", 0.1f) // used in SFSTDP:sfsInhibit

				.setValue("sfs_ff_radius", 8)
				.setValue("sfs_ff_weightAlpha", 0.001f)
				.setValue("sfs_ff_lambda", 0)

				.setValue("sfs_r_radius", 8)
				.setValue("sfs_r_weightAlpha", 0.001f)
				.setValue("sfs_R_lambda", 0) // input decay

				.setValue("hl_poolSteps", 2) // not used: see FeatureHierarchy:fhPool
				.setValue("p_alpha", 0.04f)
				.setValue("p_beta", 0.08f)
				.setValue("p_radius", 8);
		}

		// Generate the hierarchy
		std::shared_ptr<feynman::Hierarchy> h = arch.generateHierarchy();

		TextConverter textConverter(sparsity, nNeuronsChar);

		// Training time
		const int numIter = 10;

		Array2D<float> predField;
		std::string prevPredStr = "";

		// Train for a bit
		for (int iter = 0; (iter < numIter); ++iter) {
			std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

			int textLength = static_cast<int>(example.size());
			// Run through text
			for (int currentPos = 0; currentPos < textLength; currentPos += textJump) {

				// Run a simulation step of the hierarchy (learning enabled)
				const std::tuple<std::string, Array2D<float>> input = textConverter.convert(example, currentPos, windowWidth);

				const std::vector<Array2D<float>> inputVector = std::vector<Array2D<float>>{ std::get<1>(input) };
				const std::string inputStr = std::get<0>(input);

				const bool learn = true;
				h->simStep(inputVector, learn);

				prevPredStr = std::get<0>(textConverter.convert(predField));
				predField = h->getPredictions()[0];

				// show visual prediction
				if (false) plots::plotImage(predField, 3, "Prediction");

				std::cout << prevPredStr << ":" << inputStr << std::endl;
			}
		}
	}
}
