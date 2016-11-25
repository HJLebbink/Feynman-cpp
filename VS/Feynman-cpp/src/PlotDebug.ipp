#pragma once

#include <iostream>		// for cerr and cout
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>      // std::ostringstream
#include <SFML/Graphics.hpp>

#include "FixedPoint.ipp"
#include "Helpers.ipp"

using namespace feynman;

namespace plots {

	static std::map<std::string, sf::RenderWindow *> all_windows;

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
				float pixelValue = image._data_float[x + (y * hInWidth)];
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
		plotImage(sdrImg, image._size, pos, scale2, window);
	}

	void plotImage(
		const sf::Image &image,
		const int2 sizeImage,
		const float scale2,
		const std::string name)
	{
		sf::RenderWindow * window = new sf::RenderWindow();

		if (all_windows.find(name) == all_windows.end()) {
			window->create(sf::VideoMode(static_cast<unsigned int>(sizeImage.x * scale2), static_cast<unsigned int>(sizeImage.y * scale2)), name);
			all_windows[name] = window;
		}
		else {
			window = all_windows[name];
		}
		plotImage(image, sizeImage, float2{ 0.0f, 0.0f }, scale2, *window);
		window->display();
	}

	void plotImage(
		const Image2D &image,
		const float scale2,
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
		plotImage(image, { 0.0f, 0.0f }, scale2, *window);
		window->display();
	}

	void plotImage(
		const Image3D &image,
		const float scale2,
		const std::string name)
	{
		const int3 size = image.getSize();
		Image2D image2 = Image2D(int2{ size.x, size.y });
		for (int x = 0; x < size.x; ++x) {
			for (int y = 0; y < size.y; ++y) {
				write_2D(image2, x, y, read_3D(image, x, y, 0));
			}
		}
		plotImage(image2, scale2, name);
	}
}