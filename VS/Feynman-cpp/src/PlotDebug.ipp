#pragma once

#include <iostream>		// for cerr and cout
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>      // std::ostringstream
#include <SFML/Graphics.hpp>

#include "Helpers.ipp"

using namespace feynman;

namespace plots {

	void plotImage(
		const Image2D &image,
		const float2 pos,
		const float scale2,
		sf::RenderWindow &window)
	{
		const int hInWidth = image._size.x;
		const int hInHeight = image._size.y;

		sf::Image sdrImg;
		sdrImg.create(hInWidth, hInHeight);

		for (int x = 0; x < hInWidth; ++x) {
			for (int y = 0; y < hInHeight; ++y) {
				const float pixelValue = image._data[x + (y * hInWidth)];
				sf::Color c;
				c.r = c.g = c.b = static_cast<sf::Uint8>(255 * std::min(1.0f, std::max(0.0f, pixelValue)));
				sdrImg.setPixel(x, y, c);
			}
		}
		sf::Texture sdrTex;
		sdrTex.loadFromImage(sdrImg);

		sf::Sprite sprite2;
		sprite2.setTexture(sdrTex);
		sprite2.setPosition(pos.x, pos.y + (window.getSize().y - scale2 * hInHeight));
		sprite2.setScale(scale2, scale2);
		window.draw(sprite2);
	}

	static std::map<std::string, sf::RenderWindow *> all_windows;

	void plotImage(
		const Image2D &image,
		const float scale2,
		const bool rescaleContent,
		const std::string name)
	{
		sf::RenderWindow * window = new sf::RenderWindow();

		if (all_windows.find(name) == all_windows.end()) {
			window->create(sf::VideoMode(static_cast<unsigned int>(image._size.x * scale2), static_cast<unsigned int>(image._size.y * scale2)), name);
			all_windows[name] = window;
		}
		else {
			window = all_windows[name];
		}
		if (rescaleContent) {
			const float2 minmax = find_min_max(image);
			const float min = minmax.x;
			const float max = minmax.y;

			//printf("max=%f; min=%f", max, min);
			if (max > min) {
				const float diff = max - min;
				Image2D rescaledImage = Image2D(image._size);
				for (int i = 0; i < image._size.x * image._size.y; ++i) {
					rescaledImage._data[i] = ((image._data[i] - min)/diff);
				}
				plotImage(rescaledImage, { 0.0f, 0.0f }, scale2, *window);
			}
			else {
				plotImage(image, { 0.0f, 0.0f }, scale2, *window);
			}
		}
		else {
			plotImage(image, { 0.0f, 0.0f }, scale2, *window);
		}
		window->display();
	}
}