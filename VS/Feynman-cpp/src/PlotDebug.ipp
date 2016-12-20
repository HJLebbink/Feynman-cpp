#pragma once

#include <iostream>		// for cerr and cout
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>      // std::ostringstream
#include <SFML/Graphics.hpp>
#include <algorithm>


#include "FixedPoint.ipp"
#include "Helpers.ipp"

using namespace feynman;

namespace plots {

	static std::map<std::string, sf::RenderWindow *> all_windows;

	sf::RenderWindow *getWindowCache(const std::string &name, const int2 size, const float scale)
	{
		sf::RenderWindow * window = new sf::RenderWindow();

		if (all_windows.find(name) == all_windows.end()) {
			window->create(sf::VideoMode(static_cast<unsigned int>(size.x * scale), static_cast<unsigned int>(size.y * scale)), name);
			all_windows[name] = window;
		}
		else {
			window = all_windows[name];
		}
		return window;
	}

	void plotImage(
		const sf::Image &image,
		const int2 sizeImage,
		const float2 pos,
		const float scale2,
		sf::RenderWindow &window)
	{
		sf::Texture sdrTex;
		sdrTex.loadFromImage(image);

		sf::Sprite sprite2;
		sprite2.setTexture(sdrTex);
		sprite2.setPosition(pos.x, pos.y + (window.getSize().y - scale2 * sizeImage.y));
		sprite2.setScale(scale2, scale2);
		window.draw(sprite2);
	}

	void plotImage(
		const sf::Image &image,
		const int2 sizeImage,
		const float scale,
		const std::string &name)
	{
		sf::RenderWindow * window = getWindowCache(name, sizeImage, scale);
		plotImage(image, sizeImage, { 0.0f, 0.0f }, scale, *window);
		window->display();
	}

	void plotImage(
		const Array2D<float> &image,
		const float2 pos,
		const float scale,
		sf::RenderWindow &window)
	{
		const int hInWidth = image._size.x;
		const int hInHeight = image._size.y;

		if (false) {
			const float maxValue = image.getMaxValue();
			const float minValue = image.getMinValue();
			std::cout << "INFO: PlotDebug:plotImage: minValue=" << minValue << "; maxValue=" << maxValue << std::endl;

			if ((maxValue > 1.0f) || (minValue < -1.0f) || (maxValue == minValue)) {
				std::cout << "WARNING: PlotDebug:plotImage: minValue=" << minValue << "; maxValue=" << maxValue << std::endl;
				//offset = 1;
			}
		}

		sf::Image sdrImg;
		sdrImg.create(hInWidth, hInHeight);

		for (int x = 0; x < hInWidth; ++x) {
			for (int y = 0; y < hInHeight; ++y) {
				float pixelValue = image._data_float[y + (x * hInHeight)];
				sf::Color c;
				c.g = 0;
				if (pixelValue > 0) {
					c.r = static_cast<sf::Uint8>(255 * std::min(1.0f, std::max(0.0f, pixelValue)));
				}
				else {
					c.b = static_cast<sf::Uint8>(255 * std::min(1.0f, std::max(0.0f, -pixelValue)));
				}
				sdrImg.setPixel(x, y, c);
			}
		}
		plotImage(sdrImg, image._size, pos, scale, window);
	}

	void plotImage(
		const Array2D<float> &image,
		const float scale,
		const std::string &name)
	{
		const std::string name2 = name + "(" + std::to_string(image.getSize().x) + "," + std::to_string(image.getSize().y) + ")";
		sf::RenderWindow * window = getWindowCache(name2, image.getSize(), scale);
		plotImage(image, { 0.0f, 0.0f }, scale, *window);
		window->display();
	}

	void plotImage(
		const Array2D<float> &image,
		const int desiredWidth,
		const std::string &name)
	{
		const int width = std::max(image._size.x, desiredWidth);
		const float scale = static_cast<float>(width) / image._size.x;
		plotImage(image, scale, name);
	}

	void plotImage(
		const Array3D<float> &image,
		const float scale,
		const std::string &name)
	{
		const int nSlices = 1;

		const int3 size = image.getSize();
		for (int z = 0; z < std::min(nSlices, size.z); ++z) {
			Array2D<float> image2 = Array2D<float>(int2{ size.x, size.y });
			for (int x = 0; x < size.x; ++x) {
				for (int y = 0; y < size.y; ++y) {
					write_2D(image2, x, y, read_3D(image, x, y, z));
				}
			}
			plotImage(image2, scale, name + "z=" + std::to_string(z));
		}
	}

	void plotImage(
		const Array2D<float2> &image,
		const float scale,
		const std::string &name)
	{
		const int2 size = image.getSize();
		Array2D<float> image2 = Array2D<float>(size);
		for (int x = 0; x < size.x; ++x) {
			for (int y = 0; y < size.y; ++y) {
				write_2D(image2, x, y, read_2D(image, x, y).x);
			}
		}
		plotImage(image2, scale, name);
	}

}