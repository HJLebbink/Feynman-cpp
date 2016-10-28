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

#include "Plot.ipp"
#include "PlotDebug.ipp"

#include "Helpers.ipp"
#include "FeatureHierarchy.ipp"
#include "Predictor.ipp"
#include "SparseCoder.ipp"

using namespace feynman;

namespace mnist {


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

	void mnist_Anomaly_Detection()
	{
		const float moveSpeed = 1.0f; // Speed digits move across the screen
		const float spacing = 28.0f; // Spacing between digits
		const int targetLabel = 3; // Label of non-anomalous class
		const int totalMain = 3; // Amount of target (main) digits to load
		const int totalAnomalous = 2000; // Amount of anomalous digits to load
		const float anomalyRate = 0.18f; // Ratio of time which anomalies randomly appear
		const int imgPoolSize = 20; // Offscreen digit pool buffer size
		const float sensitivity = 0.1f; // Sensitivity to anomalies
		const float averageDecay = 0.01f; // Average activity decay
		const int numSuccessorsRequired = 5; // Number of successors before an anomaly is signalled
		const float spinRate = 5.0f; // How fast the digits spin
		const int okRange = 1; // Approximate range (in digits) where an anomaly flag can be compared to the actual anomaly outcome


		// Define a pseudo random number generator
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
			const float2 initWeightRange = { -0.001f, 0.001f };
			const float2 initThresholdRange = { -0.001f, 0.001f };

			sparseCoder.createRandom(scLayerDescs, hiddenSize, inhibitionRadius, initWeightRange, initThresholdRange, generator);
		}

		// --------------------------- Create the Predictor ---------------------------
		Predictor predictor;
		{
			std::vector<Predictor::PredLayerDesc> predictiveLayerDescs(4); // Predictor layer descriptors
			std::vector<FeatureHierarchy::LayerDesc> layerDescs(5); // Matching feature layer descriptors

			// Sizes
			layerDescs[0]._size = { 64, 64 };
			layerDescs[1]._size = { 48, 48 };
			layerDescs[2]._size = { 32, 32 };
			layerDescs[3]._size = { 24, 24 };

			for (size_t l = 0; l < layerDescs.size(); l++) {
				layerDescs[l]._spActiveRatio = 0.02f;
			}
			const int2 inputSize = { hInWidth, hInHeight };
			const float2 initWeightRange = { -0.01f, 0.01f };

			predictor.createRandom(inputSize, predictiveLayerDescs, layerDescs, initWeightRange, generator);
		}

		if (true) predictor.getMemoryUsage(true);

		// --------------------------- Create the Windows ---------------------------

		sf::RenderWindow window;
		window.create(sf::VideoMode(1024, 512), "MNIST Anomaly Detection");

		// --------------------------- Plot ---------------------------
		vis::Plot plot;
		const int overSizeMult = 6; // How many times the graph should extent past what is visible on-screen in terms of anomaly times
		{
			plot._curves.resize(1);
			plot._curves[0]._shadow = 0.1f;
			plot._curves[0]._name = "Prediction Error";
			plot._curves[0]._points.resize(bottomWidth * overSizeMult);

			// Initialize
			for (size_t i = 0; i < plot._curves[0]._points.size(); i++) {
				plot._curves[0]._points[i]._position = sf::Vector2f(static_cast<float>(i), 0.0f);
				plot._curves[0]._points[i]._color = sf::Color::Red;
			}
		}

		// Render target for the plot
		sf::RenderTexture plotRT;
		plotRT.create(window.getSize().y, window.getSize().y, false);

		// Resources for the plot
		sf::Texture lineGradient;
		lineGradient.loadFromFile("C:/Users/henk/OneDrive/Documents/GitHub/OgmaNeoDemos/resources/lineGradient.png");

		sf::Font tickFont;
		{
#		ifdef _WINDOWS
			tickFont.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#		else
#		ifdef __APPLE__
			tickFont.loadFromFile("/Library/Fonts/Courier New.ttf");
#		else
			tickFont.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#		endif
#		endif
		}
		// Uniform random in [0, 1]
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		// --------------------------- Digit rendering ---------------------------

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

		// Main render target
		sf::RenderTexture renderTexture;
		renderTexture.create(bottomWidth, bottomHeight);

		// Positioning values
		//const float boundingSize = static_cast<float>((bottomWidth - 28) / 2);
		//const float center = static_cast<float>(bottomWidth / 2);
		//const float minimum = center - boundingSize; //unused
		//const float maximum = center + boundingSize; //unused

		std::vector<int> mainIndices; // Indicies of main (target) class digits
		std::vector<int> anomalousIndices; // Indicies of anomalous class digits

		// Find indices for both main and anomalous
		{
			int index = 0;

			while (mainIndices.size() < totalMain || anomalousIndices.size() < totalAnomalous) {
				const int label = mnist::loadMNISTlabel(fromLabelFile, index);

				if (label == targetLabel) {
					if (mainIndices.size() < totalMain)
						mainIndices.push_back(index);
				}
				else {
					if (anomalousIndices.size() < totalAnomalous)
						anomalousIndices.push_back(index);
				}
				index++;
			}
		}

		// Sampling distributions
		std::uniform_int_distribution<int> mainDist(0, static_cast<int>(mainIndices.size()) - 1);
		std::uniform_int_distribution<int> anomolousDist(0, static_cast<int>(anomalousIndices.size()) - 1);

		// Load first digits
		std::vector<sf::Texture> imgPool(imgPoolSize);
		std::vector<float> imgSpins(imgPoolSize, 0.0f);
		std::vector<bool> imgAnomolous(imgPoolSize, false);
		std::vector<int> imgLabels(imgPoolSize, 0);

		for (size_t i = 0; i < imgPool.size(); i++) {
			mnist::Image img;
			const int index3 = mainDist(generator);

			mnist::loadMNISTimage(fromImageFile, mainIndices[index3], img);
			const int label = mnist::loadMNISTlabel(fromLabelFile, mainIndices[index3]);

			// Convert to SFML image
			sf::Image digit;
			digit.create(28, 28);

			for (unsigned int x = 0; x < digit.getSize().x; x++) {
				for (unsigned int y = 0; y < digit.getSize().y; y++) {
					int index2 = x + y * digit.getSize().x;
					sf::Color c = sf::Color::White;
					c.a = img._intensities[index2];
					digit.setPixel(x, y, c);
				}
			}
			imgPool[i].loadFromImage(digit);
			imgLabels[i] = label;
		}

		// Total width of image pool in pixels
		const float imgsSize = spacing * imgPool.size();

		// Current position
		float position = 0.0f;

		// Average anomaly score
		float averageScore = 1.0f;

		// Number of anomalously flagged successors
		int successorCount = 0;

		// Whether the previous digit was flagged as anomalous
		float prevAnomalous = 0.0f;

		// Statistical counters
		int numTruePositives = 0;
		int numFalsePositives = 0;
		int totalSamples = 0;

		// Modes
		bool quit = false;
		bool trainMode = true;

		// Simulaiton loop
		while (!quit) {

			// Poll events
			{
				sf::Event event;

				while (window.pollEvent(event)) {
					switch (event.type) {
					case sf::Event::Closed:
						quit = true;
						break;
					}
				}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
					quit = true;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
					std::cout << "INFO: setting trainMode=false" << std::endl;
					trainMode = false;
				}
			}

			if (totalSamples == 10) quit = true;

			window.clear();

			// Move digits
			position += moveSpeed;

			// If new digit needs to be loaded
			if (position > spacing) {
				// Load new digit
				sf::Texture newImg;
				mnist::Image img;

				// Load digit, get anomalous status and label
				bool anomolous = false;

				int label;

				if (trainMode) {
					int index2 = mainDist(generator);
					loadMNISTimage(fromImageFile, mainIndices[index2], img);
					label = mnist::loadMNISTlabel(fromLabelFile, mainIndices[index2]);
				}
				else {
					if (dist01(generator) < anomalyRate) {
						int index2 = anomolousDist(generator);
						loadMNISTimage(fromImageFile, anomalousIndices[index2], img);
						label = mnist::loadMNISTlabel(fromLabelFile, anomalousIndices[index2]);

						anomolous = true;
					}
					else {
						int index2 = mainDist(generator);
						loadMNISTimage(fromImageFile, mainIndices[index2], img);
						label = mnist::loadMNISTlabel(fromLabelFile, mainIndices[index2]);
					}
				}

				// Convert to SFML image
				sf::Image digit;
				digit.create(28, 28);

				for (unsigned int x = 0; x < digit.getSize().x; x++) {
					for (unsigned int y = 0; y < digit.getSize().y; y++) {
						int index2 = x + y * digit.getSize().x;
						sf::Color c = sf::Color::White;
						c.a = img._intensities[index2];
						digit.setPixel(x, y, c);
					}
				}
				newImg.loadFromImage(digit);

				// Update pool information
				imgPool.insert(imgPool.begin(), newImg);
				imgPool.pop_back();

				imgSpins.insert(imgSpins.begin(), 0.0f);
				imgSpins.pop_back();

				imgAnomolous.insert(imgAnomolous.begin(), anomolous);
				imgAnomolous.pop_back();

				imgLabels.insert(imgLabels.begin(), label);
				imgLabels.pop_back();

				totalSamples++;

				// Reset position
				position = 0.0f;
			}

			// Render to render target
			renderTexture.clear();

			const float offset = -imgsSize * 0.5f;

			for (size_t i = 0; i < imgPool.size(); i++) {
				// Add spinning motion
				imgSpins[i] += spinRate;

				sf::Sprite s;
				s.setTexture(imgPool[i]);
				s.setPosition(offset + position + i * spacing + imgPool[i].getSize().x * 0.5f, renderTexture.getSize().y * 0.5f - 14.0f + imgPool[i].getSize().y * 0.5f);
				s.setOrigin(imgPool[i].getSize().x * 0.5f, imgPool[i].getSize().y * 0.5f);
				s.setRotation(imgSpins[i]);

				renderTexture.draw(s);
			}

			renderTexture.display();

			// ------------------------------------- Anomaly detection -------------------------------------
			Image2D scInputImage = Image2D(int2{ bottomWidth, bottomHeight });
			{
				// Get image from render target
				sf::Image renderTextureImg = renderTexture.getTexture().copyToImage();

				// Copy image data
				for (unsigned int x = 0; x < renderTextureImg.getSize().x; x++) {
					for (unsigned int y = 0; y < renderTextureImg.getSize().y; y++) {
						sf::Color c = renderTextureImg.getPixel(x, y);
						scInputImage._data[x + y * renderTextureImg.getSize().x] = 0.333f * (c.r / 255.0f + c.b / 255.0f + c.g / 255.0f);
					}
				}
			}

			// Activate sparse coder
			sparseCoder.activate({ scInputImage }, 0.9f, 0.02f, generator);
			if (trainMode) {
				const float thresholdAlpha = 0.00004f;
				const float activeRatio = 0.02f;
				sparseCoder.learn(thresholdAlpha, activeRatio);
			}
			sparseCoder.stepEnd();

			// Compare (dot product)
			float anomalyScore = 0.0f;

			// Retrieve prediction
			const Image2D &newSDR_image = sparseCoder.getHiddenStates()[_back];
			const Image2D &predSDR_image = predictor.getPrediction();
			const std::vector<float> &newSDR = newSDR_image._data;
			const std::vector<float> &predSDR = predSDR_image._data;

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
						plots::plotImage(predSDR_image, 8.0f, false, "SDR Prediction");
					}
					if (true) {
						Image2D image2 = Image2D(scInputImage._size);
						std::vector<Image2D> reconstructions = { image2 };
						sparseCoder.reconstruct(predSDR_image, reconstructions);
						plots::plotImage(reconstructions.front(), 8.0f, false, "Visual Prediction");
					}
				}

				window.display();
			}
		}
	}
}