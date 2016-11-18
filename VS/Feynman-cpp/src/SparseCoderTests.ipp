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
#include "SparseCoder.ipp"

using namespace feynman;

namespace sparseCoderTests {

	// Temporary MNIST image buffer
	struct Image {
		std::vector<__int8> _intensities;
	};

	// MNIST image loading
	void loadMNISTimage(
		std::ifstream &fromFile,
		const int index,
		Image &img)
	{
		const int headerSize = 16;
		const int imageSize = 28 * 28;
		fromFile.seekg(headerSize + index * imageSize);
		if (img._intensities.size() != 28 * 28) {
			img._intensities.resize(28 * 28);
		}
		fromFile.read(reinterpret_cast<char*>(img._intensities.data()), 28 * 28);
	}

	// MNIST label loading
	int loadMNISTlabel(
		std::ifstream &fromFile,
		const int index)
	{
		const int headerSize = 8;
		fromFile.seekg(headerSize + index * 1);
		char label;
		fromFile.read(&label, 1);
		return static_cast<int>(label);
	}

	void SparseCoderTests()
	{
		std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

		// --------------------------- Create the Sparse Coder ---------------------------
		// Bottom input width and height (= Mnist data size)
		int bottomWidth = 28;
		int bottomHeight = 28;

		// Predictor hierarchy input width and height (= sparse coder size)
		int hInWidth = 32;
		int hInHeight = 32;

		SparseCoder sparseCoder;
		{
			std::vector<SparseCoder::VisibleLayerDesc> scLayerDescs(1);
			scLayerDescs[0]._size = { bottomWidth, bottomHeight };
			scLayerDescs[0]._radius = 8; // Receptive radius
			scLayerDescs[0]._weightAlpha = 0.001f;

			const int2 hiddenSize = { hInWidth, hInHeight };
			const int inhibitionRadius = 6;
			const float2 initWeightRange = { 0.0f, 0.001f };
			const float2 initThresholdRange = { 0.0f, 0.001f };

			sparseCoder.createRandom(scLayerDescs, hiddenSize, inhibitionRadius, initWeightRange, initThresholdRange, generator);
		}

		// --------------------------- Create the Windows ---------------------------

		sf::RenderWindow window;
		window.create(sf::VideoMode(1024, 512), "MNIST Anomaly Detection");

		// Load MNIST
#	ifdef _WINDOWS
		std::ifstream fromImageFile("C:/Users/henk/OneDrive/Documents/GitHub/OgmaNeoDemos/resources/train-images.idx3-ubyte", std::ios::binary | std::ios::in);
#	else
		std::ifstream fromImageFile("resources/train-images-idx3-ubyte", std::ios::binary | std::ios::in);
#	endif
		if (!fromImageFile.is_open()) {
			std::cerr << "Could not open train-images.idx3-ubyte!" << std::endl;
			return;
		}

#	ifdef _WINDOWS
		std::ifstream fromLabelFile("C:/Users/henk/OneDrive/Documents/GitHub/OgmaNeoDemos/resources/train-labels.idx1-ubyte", std::ios::binary | std::ios::in);
#	else
		std::ifstream fromLabelFile("resources/train-labels-idx1-ubyte", std::ios::binary | std::ios::in);
#	endif
		if (!fromLabelFile.is_open()) {
			std::cerr << "Could not open train-labels.idx1-ubyte!" << std::endl;
			return;
		}

		Image2D mnistDigit({ bottomWidth, bottomHeight });
		int mnistLabel;

		for (int counter = 0; counter < 80000; ++counter) {

			// Poll events
			{
				sf::Event event;

				while (window.pollEvent(event)) {
					switch (event.type) {
					case sf::Event::Closed:
						break;
					}
				}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
					break;
				}
			}

			// Load new digit
			{
				sparseCoderTests::Image img;
				sparseCoderTests::loadMNISTimage(fromImageFile, counter, img);

				// Convert to SFML image
				for (int x = 0; x < mnistDigit._size.x; ++x) {
					for (int y = 0; y < mnistDigit._size.y; ++y) {
						const int index2 = y + (x * mnistDigit._size.y);
						const unsigned int pixelChar = static_cast<unsigned int>(img._intensities[index2]);
						const float pixelFloat = (static_cast<float>(pixelChar)) / 255;
						//std::printf("INFO: (%i,%i)=%u=%f\n", x, y, pixelChar, pixelFloat);
						write_2D(mnistDigit, y, x, pixelFloat);
					}
				}
				mnistLabel = sparseCoderTests::loadMNISTlabel(fromLabelFile, counter);
				printf("INFO: i=%i: label=%i\n", counter, mnistLabel);
				plots::plotImage(mnistDigit, 8.0f, "Mnist Data");
			}

			// Activate sparse coder
			const float inputTraceDecay = 0.9f; // unused
			const float activeRatio = 0.01f;

			sparseCoder.activate({ mnistDigit }, inputTraceDecay, activeRatio, generator);
			const float thresholdAlpha = 0.00004f;
			sparseCoder.learn(thresholdAlpha, activeRatio);
			sparseCoder.stepEnd();


			const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];

			if (true) {
				plots::plotImage(newSDR_image, 8.0f, "SDR");
			}

			if (true) {
				Image2D image2 = Image2D(mnistDigit._size);
				std::vector<Image2D> reconstructions = { image2 };
				sparseCoder.reconstruct(newSDR_image, reconstructions);
				plots::plotImage(reconstructions.front(), 8.0f, "Visual SDR");
			}


			//std::this_thread::sleep_for(std::chrono::milliseconds(1000));

			/*
			// Compare (dot product)
			float anomalyScore = 0.0f;

			// Retrieve prediction
			const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];
			const Image2D &predSDR_image = predictor.getPrediction();
			const std::vector<float> &newSDR = newSDR_image._data_float;
			const std::vector<float> &predSDR = predSDR_image._data_float;

			for (size_t i = 0; i < newSDR.size(); i++) {
				anomalyScore += newSDR[i] * predSDR[i];
			}

			// Detection
			const float anomaly = (anomalyScore < (averageScore * sensitivity)) ? 1.0f : 0.0f;

			// Successor counting
			successorCount = (anomaly > 0.0f) ? (successorCount + 1) : 0;

			// If enough successors, there is a sustained anomaly that needs to be flagged
			const float sustainedAnomaly = (successorCount >= numSuccessorsRequired) ? 1.0f : 0.0f;

			// Adjust average score if in training mode
			if (trainMode) {
				averageScore = (averageDecay * averageScore) + ((1.0f - averageDecay) * anomalyScore);
			}
			// Hierarchy simulation step
			predictor.simStep(newSDR_image, newSDR_image, generator, trainMode);

			// Shift plot y values
			for (int i = static_cast<int>(plot._curves[0]._points.size()) - 1; i >= 1; i--) {
				plot._curves[0]._points[i]._position.y = plot._curves[0]._points[i - 1]._position.y;
			}

			// Add anomaly to plot
			plot._curves[0]._points.front()._position.y = sustainedAnomaly;

			// See if an anomaly is in range
			const int center2 = imgPoolSize / 2;

			// Gather statistics
			bool anomalyInRange = false;

			for (int dx = -okRange; dx <= okRange; dx++) {
				if (imgAnomolous[center2 + dx]) {
					anomalyInRange = true;
					break;
				}
			}
			// First detection
			bool firstDetection = (prevAnomalous == 0.0f) && (sustainedAnomaly == 1.0f);

			if (!trainMode && firstDetection) {
				if (anomalyInRange)
					numTruePositives++;
				else
					numFalsePositives++;
			}

			prevAnomalous = sustainedAnomaly;

			// Rendering of gui stuff
			if (true) {
				// show the digits
				if (true) {
					sf::Sprite s;
					s.setTexture(renderTexture.getTexture());
					float scale = window.getSize().y / static_cast<float>(bottomHeight);
					s.setScale(scale, scale);
					window.draw(s);
				}

				// Show pool chain
				if (true) {
					const float miniChainSpacing = 24.0f;
					for (size_t i = 0; i < imgPool.size(); i++) {
						sf::RectangleShape rs;
						rs.setPosition(i * miniChainSpacing + 4.0f, 4.0f);

						sf::Text text;
						text.setFont(tickFont);
						text.setCharacterSize(24);
						text.setString(std::to_string(imgLabels[i]));
						text.setPosition(rs.getPosition() + sf::Vector2f(2.0f, -2.0f));
						rs.setSize(sf::Vector2f(miniChainSpacing, miniChainSpacing));
						rs.setFillColor((firstDetection) ? sf::Color::Red : sf::Color::Green);

						window.draw(rs);
						window.draw(text);
					}
					{	// Show statistics
						sf::Text text;
						text.setFont(tickFont);
						text.setCharacterSize(16);
						text.setString("Total samples: " + std::to_string(totalSamples));
						text.setPosition(4.0f, 32.0f);
						window.draw(text);
						text.setString("True Positives: " + std::to_string(numTruePositives));
						text.setPosition(4.0f, 32.0f + 32.0f);
						window.draw(text);
						text.setString("False Positives: " + std::to_string(numFalsePositives));
						text.setPosition(4.0f, 32.0f + 32.0f + 32.0f);
						window.draw(text);
					}
				}

				// Render plot of the anamaly to the right
				if (true) {
					plotRT.clear(sf::Color::White);
					plot.draw(
						plotRT,
						lineGradient,
						tickFont,
						0.5f,
						sf::Vector2f(0.0f, static_cast<float>(bottomWidth * overSizeMult)),
						sf::Vector2f(-1.0f, 2.0f),
						sf::Vector2f(50.0f, 50.0f),
						sf::Vector2f(50.0f, 4.0f),
						2.0f,
						2.0f,
						2.0f,
						5.0f,
						4.0f,
						3
					);
					plotRT.display();

					// Anomaly flag
					if (sustainedAnomaly > 0.0f) {
						sf::RectangleShape rs;
						rs.setFillColor(sf::Color::Red);
						rs.setSize(sf::Vector2f(512.0f, 12.0f));
						window.draw(rs);
					}

					// Show plot
					sf::Sprite plotS;
					plotS.setTexture(plotRT.getTexture());
					plotS.setPosition(512.0f, 0.0f);
					window.draw(plotS);
				}

				// Show SDRs is corner bottom left
				if (true) {
					if (true) {
						//const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];
						plots::plotImage(newSDR_image, { 0.0f, 0.0f }, 2.0f, window);
					}
					if (false) {
						//const Image2D &predSDR_image = predictor.getPrediction();
						plots::plotImage(predSDR_image, 8.0f, "SDR Prediction");
					}
					if (true) {
						Image2D image2 = Image2D(scInputImage._size);
						std::vector<Image2D> reconstructions = { image2 };
						sparseCoder.reconstruct(predSDR_image, reconstructions);
						plots::plotImage(reconstructions.front(), 8.0f, "Visual Prediction");
					}
				}

				window.display();
			}

			*/
		}
	}
}