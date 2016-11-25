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
		std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

		// Uniform distribution in [0, 1]
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		sf::RenderWindow window;
		window.create(sf::VideoMode(800, 800), "Video Test", sf::Style::Default);

		// Uncap framerate
		window.setFramerateLimit(0);

		// Target file name
		const std::string fileName = "C:/Users/henk/OneDrive/Documents/GitHub/OgmaNeoDemos/resources/Tesseract.wmv";
		//const std::string fileName = "C:/Data/Tesseract.wmv";
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

		// --------------------------- Create the Hierarchy ---------------------------

		feynman::Architect arch;
		arch.initialize(1234);

		// 3 input layers for RGB
		arch.addInputLayer(int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) })
			.setValue("in_p_alpha", 0.02f)
			.setValue("in_p_radius", 8);

		arch.addInputLayer(int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) })
			.setValue("in_p_alpha", 0.02f)
			.setValue("in_p_radius", 8);

		arch.addInputLayer(int2{ static_cast<int>(rescaleRT.getSize().x), static_cast<int>(rescaleRT.getSize().y) })
			.setValue("in_p_alpha", 0.02f)
			.setValue("in_p_radius", 8);

		for (int l = 0; l < 1; l++)
			arch.addHigherLayer(int2{ 64, 64 }, feynman::_chunk)
			.setValue("sfc_chunkSize", int2{ 6, 6 })
			.setValue("sfc_ff_radius", 8)
			.setValue("hl_poolSteps", 2)
			.setValue("sfc_numSamples", 2)
			.setValue("sfc_weightAlpha", 0.01f)
			.setValue("sfc_biasAlpha", 0.1f)
			.setValue("sfc_gamma", 0.92f)
			.setValue("p_alpha", 0.04f)
			.setValue("p_beta", 0.08f)
			.setValue("p_radius", 8);

		// 4 layers using chunk encoders
		for (int l = 0; l < 0; l++)
			arch.addHigherLayer({ 64, 64 }, feynman::_stdp)
			.setValue("sfs_ff_radius", 8)
			.setValue("hl_poolSteps", 2)
			.setValue("sfs_inhibitionRadius", 6)
			.setValue("sfs_activeRatio", 0.02f)
			.setValue("sfs_weightAlpha", 0.01f)
			.setValue("sfs_biasAlpha", 0.1f)
			.setValue("sfs_gamma", 0.92f)
			.setValue("p_alpha", 0.04f)
			.setValue("p_beta", 0.08f)
			.setValue("p_radius", 8);

		// Generate the hierarchy
		std::shared_ptr<feynman::Hierarchy> h = arch.generateHierarchy();

		// Input and prediction fields for color components
		Image2D inputFieldR(rescaleRT.getSize().x, rescaleRT.getSize().y);
		Image2D inputFieldG(rescaleRT.getSize().x, rescaleRT.getSize().y);
		Image2D inputFieldB(rescaleRT.getSize().x, rescaleRT.getSize().y);
		Image2D predFieldR(rescaleRT.getSize().x, rescaleRT.getSize().y);
		Image2D predFieldG(rescaleRT.getSize().x, rescaleRT.getSize().y);
		Image2D predFieldB(rescaleRT.getSize().x, rescaleRT.getSize().y);

		// Unit Gaussian noise for input corruption
		std::normal_distribution<float> noiseDist(0.0f, 1.0f);

		// Training time
		const int numIter = 4;

		// UI update resolution
		const int progressBarLength = 40;
		const int progressUpdateTicks = 4;
		bool quit = false;

		// Train for a bit
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

				// Convert to SFML image
				sf::Image img;
				{
					img.create(frame.cols, frame.rows);
					for (unsigned int x = 0; x < img.getSize().x; x++) {
						for (unsigned int y = 0; y < img.getSize().y; y++) {
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
							sf::Color c = reImg.getPixel(x, y);

							write_2D(inputFieldR, x, y, c.r / 255.0f * (1.0f - blendPred) + read_2D(predFieldR, x, y) * blendPred);
							write_2D(inputFieldG, x, y, c.r / 255.0f * (1.0f - blendPred) + read_2D(predFieldG, x, y) * blendPred);
							write_2D(inputFieldB, x, y, c.r / 255.0f * (1.0f - blendPred) + read_2D(predFieldB, x, y) * blendPred);
						}
					}
				}

				std::vector<Image2D> inputVector = { inputFieldR, inputFieldG, inputFieldB };

				// Run a simulation step of the hierarchy (learning enabled)
				const bool learn = true;
				h->simStep(inputVector, learn);

				predFieldR = h->getPredictions()[0];
				predFieldG = h->getPredictions()[1];
				predFieldB = h->getPredictions()[2];


				// show visual prediction
				if (true) {

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

					plots::plotImage(img, predFieldR._size, 3, "Prediction");
				}

				// Show progress bar
				float ratio = static_cast<float>(currentFrame + 1) / captureLength;

				// Console
				{
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

			std::vector<Image2D> inputVector = { predFieldR, predFieldG, predFieldB };
			h->simStep(inputVector, false);

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