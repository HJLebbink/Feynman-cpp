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

const std::vector<std::string> data_Y = {
	"xx......xx",
	"xxx....xxx",
	".xxx..xxx.",
	"..xxxxxx..",
	"...xxxx...",
	"...xxxx...",
	"....xx....",
	"....xx....",
	"....xx....",
	"....xx...."
};


Image2D convert(const std::vector<std::string> &data) {
	Image2D result = Image2D(int2{ static_cast<int>(data[0].length()), static_cast<int>(data.size())});
	for (int x = 0; x < static_cast<int>(data.size()); ++x) {
		const std::string str = data[x];
		for (int y = 0; y < static_cast<int>(str.length()); ++y) {
			write_2D(result, x, y, (str[y] == 'x') ? 1.0f : 0.0f);
		}
	}
	return result;
}

void recallTest_AAAX() {

	std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));


	// Create input images
	std::vector<Image2D> inputImages = std::vector<Image2D>();
	{
		if (true) {
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A)); // confused with Y
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_X));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_Y));
		}
		if (false) {
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_A));
			inputImages.push_back(convert(data_X));
		}
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
		const int c = 8;

		layerDescs[0]._size = { 3*c, 3*c };
		layerDescs[1]._size = { 2*c, 2*c };
		layerDescs[2]._size = { c, c };

		for (size_t layer = 0; layer < layerDescs.size(); layer++) {
			layerDescs[layer]._recurrentRadius = 6;
			layerDescs[layer]._spActiveRatio = 0.04f;
			layerDescs[layer]._spBiasAlpha = 0.01f;

			pLayerDescs[layer]._alpha = 0.08f;
			pLayerDescs[layer]._beta = 0.16f;
		}
	}

	// Create predictive hierarchy
	Predictor predictor;
	{
		const float2 initWeightRange = { -0.01f, 0.01f };
		predictor.createRandom(hiddenSize, pLayerDescs, layerDescs, initWeightRange, generator);
	}


	sf::RenderWindow window;
	window.create(sf::VideoMode(3 * 10 * 32, 1 * 10 * 32), "Recall Test AAAX");


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

		const Image2D &inputImage = inputImages[counter % inputImages.size()];

		// Activate sparse coder
		{
			sparseCoder.activate({ inputImage }, 0.9f, 0.02f, generator);
			if (trainMode) {
				const float thresholdAlpha = 0.0004f;// 0.00004f;
				const float activeRatio = 0.02f;
				sparseCoder.learn(thresholdAlpha, activeRatio);
			}
			sparseCoder.stepEnd();
		}

		// Retrieve prediction
		const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];
		const Image2D &predSDR_image = predictor.getPrediction();

		Image2D predictedImage;
		{
			Image2D image2 = Image2D(inputImage._size);
			std::vector<Image2D> reconstructions = { image2 };
			sparseCoder.reconstruct(predSDR_image, reconstructions);
			predictedImage = reconstructions.front();
		}

		Image2D reconstructionErrorImage = Image2D(inputImage._size);
		{	// calculate the prediction error
			float pixelMissmatch = 0;
			for (size_t i = 0; i < inputImage._data_float.size(); ++i) {
				const float predictedPixel = std::min(1.0f, std::max(0.0f, predictedImage._data_float[i]));
				const float errorPerPixel = abs(inputImage._data_float[i] - predictedPixel);
				if (errorPerPixel > 1.0) {
					printf("WARNING: errorPerPixel %f, input=%f, predicted=%f\n", errorPerPixel, inputImage._data_float[i], predictedImage._data_float[i]);
				}

				reconstructionErrorImage._data_float[i] = errorPerPixel;
				pixelMissmatch += errorPerPixel;
			}
			pixelMissmatch = pixelMissmatch / inputImage._data_float.size();
			printf("trainstep %i: pixelMissmatch %f\n", counter, pixelMissmatch);
		}

		// plot stuff
		{
			window.clear();
			plots::plotImage(inputImage, float2{ 0 * 10.0f * 32.0f, 0.0f }, 32.0f, window);
			plots::plotImage(predictedImage, float2{ 1 * 10.0f * 32.0f, 0.0f }, 32.0f, window);
			plots::plotImage(reconstructionErrorImage, float2{ 2 * 10.0f * 32.0f, 0.0f }, 32.0f, window);
			window.display();

			plots::plotImage(newSDR_image, 8.0f, "Current SDR");
			plots::plotImage(predSDR_image, 8.0f, "Predicted SDR");
		}

		// Hierarchy simulation step
		predictor.simStep(newSDR_image, newSDR_image, generator, trainMode);

		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
		counter++;
	}
}