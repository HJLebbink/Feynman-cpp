// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <random>
#include <algorithm>


#include "Helpers.ipp"
#include "Architect.ipp"
#include "Hierarchy.ipp"
#include "SparseFeaturesSTDP.ipp"


using namespace feynman;
using namespace cv;

namespace video {

	void video_Prediction() {

		// Initialize a random number generator
		//std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));
		std::mt19937 generator(0xDEADBEEF);


		// Uniform distribution in [0, 1]
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		sf::RenderWindow window;
		window.create(sf::VideoMode(800, 800), "Video Test", sf::Style::Default);

		// Uncap framerate
		window.setFramerateLimit(0);

		// Target file name
		const std::string fileName = "C:/Data/Tesseract.wmv";
		//const std::string fileName = "C:/Data/The.History.Of.Sega.Dreamcast.(2004).wmv";
		


		//const std::string fileName = "C:/Data/Tesseract.wmv";
		sf::Font font;

#		if defined(_WINDOWS)
		font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#		else
#		ifdef __APPLE__
		font.loadFromFile("/Library/Fonts/Courier New.ttf");
#		else
		font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#		endif
#		endif

		// Parameters
		const int frameSkip = 3; // Frames to skip
		const float videoScale = 1.0f; // Rescale ratio

		// Video rescaling render target
		sf::RenderTexture rescaleRT;
		rescaleRT.create(2*64, 2*64);

		// --------------------------- Create the Hierarchy ---------------------------

		feynman::Architect arch;
		arch.initialize(1234);

		const int2 inputLayerSize = int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) };
		{
			// 3 input layers for RGB
			arch.addInputLayer(inputLayerSize)
				.setValue("in_p_alpha", 0.02f)
				.setValue("in_p_radius", 8);
			
			arch.addInputLayer(inputLayerSize)
				.setValue("in_p_alpha", 0.02f)
				.setValue("in_p_radius", 8);

			arch.addInputLayer(inputLayerSize)
				.setValue("in_p_alpha", 0.02f)
				.setValue("in_p_radius", 8);
				
		}
		{
			for (int l = 0; l < 3; l++)
				arch.addHigherLayer(int2{ 64 - l, 64 - l }, feynman::_old)
				.setValue("inhibitionRadius", 5)
				.setValue("activeRatio", 0.02f)
				.setValue("biasAlpha", 0.02f)
				.setValue("initWeightRange", float2{ 0.0, 1.0 })
				.setValue("initBiasRange", float2{ -0.01, 0.01 })


				.setValue("ff_radius", 20)
				.setValue("ff_weightAlpha", 0.025f) // used in SparseFeaturesChunk:sfcLearnWeights 

				.setValue("r_radius", 8)
				.setValue("r_weightAlpha", 0.025f)

				.setValue("hl_poolSteps", 4) // used in FeatureHierarchy:fhPool
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
				arch.addHigherLayer({ 64-l, 64-l }, feynman::_stdp)
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

		// Input and prediction fields for color components
		Array2D<float> inputFieldR(inputLayerSize);
		Array2D<float> inputFieldG(inputLayerSize);
		Array2D<float> inputFieldB(inputLayerSize);
		Array2D<float> predFieldR(inputLayerSize);
		Array2D<float> predFieldG(inputLayerSize);
		Array2D<float> predFieldB(inputLayerSize);

		// Unit Gaussian noise for input corruption
		std::normal_distribution<float> noiseDist(0.0f, 1.0f);

		// Training time
		const int numIter = 60;
		const int nFrames = 5000;


		// UI update resolution
		const int progressBarLength = 40;
		const int progressUpdateTicks = 4;
		bool quit = false;

		// Train for a bit
		
		int counter = 0;
		for (int iter = 0; ((iter < numIter) && !quit); ++iter) {
			std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

			// Open the video file
			VideoCapture capture(fileName);
			Mat frame;

			if (!capture.isOpened()) {
				std::cerr << "Could not open capture: " << fileName << std::endl;
			}
			std::cout << "Running through capture: " << fileName << std::endl;
			const int captureLength = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_COUNT));
			int currentFrame = 0;

			// Run through video
			do {
				// Read several discarded frames if frame skip is > 0
				for (int i = 0; i < frameSkip; i++) {
					capture >> frame;
					currentFrame++;
					if (frame.empty())
						break;
				}

				if (frame.empty())
					break;

				counter++;
				if (counter > nFrames) {
					iter = numIter;
				}

				std::cout << "INFO: done " << counter << " steps" << std::endl;


				// Convert to SFML image
				sf::Image img;
				{
					img.create(frame.cols, frame.rows);
					for (unsigned int x = 0; x < img.getSize().x; ++x) {
						for (unsigned int y = 0; y < img.getSize().y; ++y) {
							sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 2];
							sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
							sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 0];

							img.setPixel(x, y, sf::Color(r, g, b));
						}
					}
				}

				// To SFML texture
				sf::Texture tex;
				{
					tex.loadFromImage(img);
					tex.setSmooth(true);
				}

				// Rescale using render target
				{
					sf::Sprite s;

					s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);
					s.setTexture(tex);
					s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
					const float scale = videoScale * std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);
					s.setScale(scale, scale);
					rescaleRT.clear();
					rescaleRT.draw(s);
					rescaleRT.display();
				}

				// SFML image from rescaled frame
				{
					sf::Image reImg = rescaleRT.getTexture().copyToImage();

					// Get input buffers
					for (unsigned int x = 0; x < reImg.getSize().x; x++) {
						for (unsigned int y = 0; y < reImg.getSize().y; y++) {
							const sf::Color c = reImg.getPixel(x, y);

							write_2D(inputFieldR, x, y, c.r / 255.0f);
							write_2D(inputFieldG, x, y, c.g / 255.0f);
							write_2D(inputFieldB, x, y, c.b / 255.0f);
						}
					}
					plots::plotImage(reImg, {static_cast<int>(reImg.getSize().x), static_cast<int>(reImg.getSize().y) }, 4, "data");
					//plots::plotImage(inputFieldR, 4, "dataR");
					//plots::plotImage(inputFieldG, 4, "dataG");
					//plots::plotImage(inputFieldB, 4, "dataB");
				}

				// Run a simulation step of the hierarchy (learning enabled)
				std::vector<Array2D<float>> inputVector = { inputFieldR, inputFieldG, inputFieldB };
				//std::vector<Array2D<float>> inputVector = { inputFieldR };
				const bool learn = true;
				if (EXPLAIN) std::cout << "EXPLAIN: Video_Prediction: starting simstep on 3 inputs." << std::endl;

				h->simStep(inputVector, learn);

				predFieldR = h->getPredictions()[0];
				predFieldG = h->getPredictions()[1];
				predFieldB = h->getPredictions()[2];

				// show visual prediction
				if (true) {
					plots::plotImage(predFieldR, 3, "PredictionR");
					//plots::plotImage(predFieldG, 3, "PredictionG");
					//plots::plotImage(predFieldB, 3, "PredictionB");

					sf::Image prediction_img;
					prediction_img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);
					for (size_t x = 0; x < rescaleRT.getSize().x; x++) {
						for (size_t y = 0; y < rescaleRT.getSize().y; y++) {
							sf::Color c;
							c.r = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldR, x, y)));
							//c.g = c.b = c.r;
							c.g = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldG, x, y)));
							c.b = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldB, x, y)));
							prediction_img.setPixel(x, y, c);
						}
					}
					plots::plotImage(prediction_img, predFieldR._size, 3, "Prediction");
				}

				// Show progress bar
				float ratio = static_cast<float>(currentFrame + 1) / captureLength;

				// show progress percentage on console
				if (false) { 
					if (currentFrame % progressUpdateTicks == 0) {
						std::cout << "\r";
						std::cout << "[";

						int bars = static_cast<int>(std::round(ratio * progressBarLength));
						int spaces = progressBarLength - bars;
						for (int i = 0; i < bars; i++)
							std::cout << "=";

						for (int i = 0; i < spaces; i++)
							std::cout << " ";

						std::cout << "] " << static_cast<int>(ratio * 100.0f) << "%";
					}
				}

				// UI
				if (currentFrame % progressUpdateTicks == 0) {
					sf::Event windowEvent;

					while (window.pollEvent(windowEvent))
					{
						switch (windowEvent.type)
						{
						case sf::Event::Closed:
							quit = true;
							break;
						default:
							break;
						}
					}

					if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
						quit = true;

					// Show progress bar
					window.clear();
					sf::RectangleShape rs;
					rs.setPosition(8.0f, 8.0f);
					rs.setSize(sf::Vector2f(128.0f * ratio, 32.0f));
					rs.setFillColor(sf::Color::Red);
					window.draw(rs);
					rs.setFillColor(sf::Color::Transparent);
					rs.setOutlineColor(sf::Color::White);
					rs.setOutlineThickness(2.0f);
					rs.setSize(sf::Vector2f(128.0f, 32.0f));
					window.draw(rs);
					sf::Text t;
					t.setFont(font);
					t.setCharacterSize(20);
					std::string st;
					st += std::to_string(static_cast<int>(ratio * 100.0f)) + "% (pass " + std::to_string(iter + 1) + " of " + std::to_string(numIter) + ")";
					t.setString(st);
					t.setPosition(144.0f, 8.0f);
					t.setFillColor(sf::Color::White);
					window.draw(t);
					window.display();
				}

			} while (!frame.empty() && !quit);

			// Make sure bar is at 100%
			std::cout << "\r" << "[";
			for (int i = 0; i < progressBarLength; i++) {
				std::cout << "=";
			}
			std::cout << "] 100%" << std::endl;
		}

		// ---------------------------- Presentation Simulation Loop -----------------------------

		if (false) {
			// feature extraction tests

			//Array2D<float> hiddenState = Array2D<float>({ 64.0f, 64.0f });
			//hiddenState.fill(0.0f);

			//int counter = 0;

			//for (int x = 0; x < 64; ++x) {
			//	for (int y = 0; y < 64; ++y) {
			//		hiddenState.set(x, y, 3.0f);
			//		Array2D<float> feature = h->getFeature(hiddenState);
			//		const float activationSum = feature.sum();
			//		//std::cout << "INFO: activation sum (" << x << "," << y << ")=" << activationSum << std::endl;
			//		if (activationSum > 400) {
			//			feature.set(x, y, -1);
			//			plots::plotImage(feature, DEBUG_IMAGE_WIDTH, "feature(" + std::to_string(x) + "," + std::to_string(y)+")");
			//		}
			//		hiddenState.set(x, y, 0.0f);
			//		counter++;
			//		if (counter > 10) break;
			//	}
			//}
		}


		window.setVerticalSyncEnabled(true);

		do {
			// ----------------------------- Input -----------------------------
			{
				sf::Event windowEvent;

				while (window.pollEvent(windowEvent))
				{
					switch (windowEvent.type)
					{
					case sf::Event::Closed:
						quit = true;
						break;
					default:
						break;
					}
				}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
					quit = true;
			}
			
			window.clear();

			std::vector<Array2D<float>> inputVector = { predFieldR, predFieldG, predFieldB };
			//std::vector<Array2D<float>> inputVector = { predFieldR };
			const bool learn = false;
			h->simStep(inputVector, learn);

			predFieldR = h->getPredictions()[0];
			predFieldG = h->getPredictions()[1];
			predFieldB = h->getPredictions()[2];

			sf::Image img;
			img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

			for (size_t x = 0; x < rescaleRT.getSize().x; x++) {
				for (size_t y = 0; y < rescaleRT.getSize().y; y++) {
					sf::Color c;
					c.r = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldR, x, y)));
					c.g = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldG, x, y)));
					c.b = 255.0f * std::min(1.0f, std::max(0.0f, read_2D(predFieldB, x, y)));
					img.setPixel(x, y, c);
				}
			}

			sf::Texture tex;
			tex.loadFromImage(img);
			sf::Sprite s;
			s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);
			s.setTexture(tex);
			s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
			float scale = videoScale * std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);
			s.setScale(sf::Vector2f(scale, scale));

			window.clear();
			window.draw(s);
			window.display();
		} while (!quit);
	}
}