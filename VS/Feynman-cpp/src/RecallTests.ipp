#pragma once

#include <iostream>		// for cerr and cout
#include <random>
#include <chrono>
#include <thread>

#include "Plot.ipp"
#include "PlotDebug.ipp"

#include "Helpers.ipp"
#include "SparseCoder.ipp"
#include "Predictor.ipp"

using namespace feynman;


Image2D convert(const std::vector<std::string> &data) {
	Image2D result = Image2D(int2{ static_cast<int>(data[0].length()), static_cast<int>(data.size())});
	for (size_t x = 0; x < data.size(); ++x) {
		const std::string str = data[x];
		for (size_t y = 0; y < str.length(); ++y) {
			write_2D(result, x, y, (str[y] == 'x') ? 1.0f : 0.0f);
		}
	}
	return result;
}

void recallTest_AAAX() {

	std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

	// Create input images
	std::vector<Image2D> inputImages = std::vector<Image2D>(4);
	{
		const std::vector<std::string> data_A = {
			"....xx....",
			"....xx....",
			"...xxxx...",
			"...xxxx...",
			"..xx..xx..",
			"..xx..xx..",
			".xxxxxxxx.",
			".xxxxxxxx.",
			"xxx....xxx",
			"xx......xx"
		};

		const std::vector<std::string> data_X = {
			".x......x.",
			"xxx....xxx",
			".xxx..xxx.",
			"..xxxxxx..",
			"...xxxx...",
			"...xxxx...",
			"..xxxxxx..",
			".xxx..xxx.",
			"xxx....xxx",
			".x......x."
		};

		inputImages[0] = convert(data_A);
		inputImages[1] = convert(data_A);
		inputImages[2] = convert(data_A);
		inputImages[3] = convert(data_X);
	}

	// Create the Sparse Coder
	SparseCoder sparseCoder;
	const int2 hiddenSize = { 32, 32 };
	{
		std::vector<SparseCoder::VisibleLayerDesc> scLayerDescs(1);
		scLayerDescs[0]._size = inputImages[0]._size;
		scLayerDescs[0]._radius = 8; // Receptive radius
		scLayerDescs[0]._weightAlpha = 0.001f;

		const int inhibitionRadius = 8;
		const float2 initWeightRange = { -0.001f, 0.001f };
		const float2 initThresholdRange = { -0.001f, 0.001f };

		sparseCoder.createRandom(scLayerDescs, hiddenSize, inhibitionRadius, initWeightRange, initThresholdRange, generator);
	}

	// Create hierarchy structure
	std::vector<FeatureHierarchy::LayerDesc> layerDescs(3);
	std::vector<Predictor::PredLayerDesc> pLayerDescs(3);
	{
		layerDescs[0]._size = { 48, 48 };
		layerDescs[1]._size = { 32, 32 };
		layerDescs[2]._size = { 24, 24 };

		for (size_t layer = 0; layer < layerDescs.size(); layer++) {
			layerDescs[layer]._recurrentRadius = 6;
			layerDescs[layer]._spActiveRatio = 0.02f;
			layerDescs[layer]._spBiasAlpha = 0.01f;

			pLayerDescs[layer]._alpha = 0.04f;
			pLayerDescs[layer]._beta = 0.04f;
		}
	}

	// Create predictive hierarchy
	Predictor predictor;
	{
		const float2 initWeightRange = { -0.01f, 0.01f };
		predictor.createRandom(hiddenSize, pLayerDescs, layerDescs, initWeightRange, generator);
	}


	sf::RenderWindow window;
	window.create(sf::VideoMode(10 * 32 * 2, 10 * 32 * 1), "Recall Test AAAX");


	int counter = 0;
	bool trainMode = true;
	bool quit = false;
	while (!quit) {

		// Poll events
		{
			sf::Event myEvent;
			while (window.pollEvent(myEvent)) {
				switch (myEvent.type) {
				case sf::Event::Closed:
					quit = true;
					break;
				}
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
				quit = true;
			}
		}

		const int inputIndex = counter % inputImages.size();
		const Image2D &inputImage = inputImages[inputIndex];

		// Activate sparse coder
		{
			sparseCoder.activate({ inputImage }, 0.9f, 0.01f, generator);
			if (trainMode) {
				const float thresholdAlpha = 0.004f;// 0.00004f;
				const float activeRatio = 0.01f;
				sparseCoder.learn(thresholdAlpha, activeRatio);
			}
			sparseCoder.stepEnd();
		}

		// Retrieve prediction
		const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];
		const Image2D &predSDR_image = predictor.getPrediction();

		// Hierarchy simulation step
		predictor.simStep(newSDR_image, newSDR_image, generator, trainMode);

		// plot stuff
		{
			window.clear();
			plots::plotImage(inputImage, float2{ 0.0f, 0.0f }, 32.0f, window);

			if (true) {
				Image2D image2 = Image2D(inputImage._size);
				std::vector<Image2D> reconstructions = { image2 };
				sparseCoder.reconstruct(predSDR_image, reconstructions);
				plots::plotImage(reconstructions.front(), float2{ 10.0f * 32.0f, 0.0f }, 32.0f, window);
			}

			plots::plotImage(newSDR_image, 8.0f, false, "Current SDR");
			plots::plotImage(predSDR_image, 8.0f, false, "Predicted SDR");

			window.display();
		}

		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
		counter++;
	}
}