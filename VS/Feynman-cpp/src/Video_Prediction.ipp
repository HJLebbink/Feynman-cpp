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

#include "Helpers.ipp"
#include "Predictor.ipp"

using namespace feynman;
using namespace cv;

namespace video {

	void video_Prediction() {

		// Initialize a random number generator
		std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

		// Uniform distribution in [0, 1]
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		sf::RenderWindow window;
		window.create(sf::VideoMode(800, 800), "Video Test", sf::Style::Default);

		// Uncap framerate
		window.setFramerateLimit(0);

		// Target file name
		std::string fileName = "C:/Users/henk/OneDrive/Documents/GitHub/OgmaNeoDemos/resources/Tesseract.wmv";
		sf::Font font;

#		ifdef _WINDOWS
		font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#		else
#		ifdef __APPLE__
		font.loadFromFile("/Library/Fonts/Courier New.ttf");
#		else
		font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#		endif
#		endif

		// Parameters
		const int frameSkip = 4; // Frames to skip
		const float videoScale = 1.0f; // Rescale ratio
		const float blendPred = 0.0f; // Ratio of how much prediction to blend in to input (part of input corruption)

		// Video rescaling render target
		sf::RenderTexture rescaleRT;
		rescaleRT.create(128, 128);

		// --------------------------- Create the Sparse Coder ---------------------------
		// Input images
		Image2D inputImage = Image2D(int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) });
		Image2D inputImageCorrupted = Image2D(int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) });

		// Predictive hierarchy
		Predictor predictor;
		{
			// Hierarchy structure
			std::vector<FeatureHierarchy::LayerDesc> layerDescs(3);
			std::vector<Predictor::PredLayerDesc> pLayerDescs(3);

			layerDescs[0]._size = { 64, 64 };
			layerDescs[1]._size = { 64, 64 };
			layerDescs[2]._size = { 64, 64 };

			for (size_t l = 0; l < layerDescs.size(); l++) {
				layerDescs[l]._recurrentRadius = 6;
				layerDescs[l]._spActiveRatio = 0.04f;
				layerDescs[l]._spBiasAlpha = 0.01f;

				pLayerDescs[l]._alpha = 0.08f;
				pLayerDescs[l]._beta = 0.16f;
			}
			predictor.createRandom({ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) }, pLayerDescs, layerDescs, { 0.0f, 0.01f }, generator);
			if (true) predictor.getMemoryUsage(true);
		}

		// Host image buffer
		std::vector<float> pred(rescaleRT.getSize().x * rescaleRT.getSize().y, 0.0f); // init with content equal to zero

		// Unit Gaussian noise for input corruption
		std::normal_distribution<float> noiseDist(0.0f, 1.0f);

		// Training time
		const int numIter = 10;

		// UI update resolution
		const int progressBarLength = 40;
		const int progressUpdateTicks = 4;
		bool quit = false;

		// Train for a bit
		for (int iter = 0; iter < numIter && !quit; iter++) {
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

				// Convert to SFML image
				sf::Image img;
				{
					img.create(frame.cols, frame.rows);

					for (unsigned int x = 0; x < img.getSize().x; x++) {
						for (unsigned int y = 0; y < img.getSize().y; y++) {
							sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 0];
							sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
							sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 2];

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
							sf::Color c = reImg.getPixel(x, y);
							const float mono = (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f) * 0.3333f;
							const float blend = blendPred * pred[x + y * reImg.getSize().x] + (1.0f - blendPred) * mono;
							//inputImage._data[x + y * reImg.getSize().x] = mono;
							write_2D(inputImage, y, x, mono);
							//inputImageCorrupted._data[x + y * reImg.getSize().x] = blend;
							write_2D(inputImageCorrupted, y, x, blend);
						}
					}
				}

				// Run a simulation step of the hierarchy (learning enabled)
				predictor.simStep(inputImage, inputImageCorrupted, generator, true);

				// Get the resulting prediction (for prediction blending)
				copy(predictor.getPrediction(), pred);

				// Show progress bar
				float ratio = static_cast<float>(currentFrame + 1) / captureLength;
				{
					// Console
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

				// plot visual prediction
				if (true) {
					plots::plotImage(predictor.getPrediction(), 4.0f, "Visual Prediction");
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
					}
				}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
					quit = true;
			}
			
			// Write prediction as input
			copy(pred, inputImage);

			// Run a simulation step with learning disabled
			predictor.simStep(inputImage, inputImage, generator, false);

			// Display prediction
			copy(predictor.getPrediction(), pred);

			sf::Image img;
			img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

			for (unsigned int x = 0; x < rescaleRT.getSize().x; x++) {
				for (unsigned int y = 0; y < rescaleRT.getSize().y; y++) {
					sf::Color c;
					c.r = c.g = c.b = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, pred[x + y * img.getSize().x])));
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