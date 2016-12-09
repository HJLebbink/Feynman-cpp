#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include "Helpers.ipp"
#include "PlotDebug.ipp"

namespace feynman {

	class TextConverter {

	private:

		std::unordered_map<char, std::vector<float>> _data;
		int _nNeurons;
		int _nNeuronsActive;
		int _sparsity;

	public:

		TextConverter(const float sparsity, const int nNeurons) {
			_data = std::unordered_map<char, std::vector<float>>();
			_nNeurons = nNeurons;
			_sparsity = sparsity;
			_nNeuronsActive = static_cast<int>(std::round(nNeurons * sparsity));

			initRandom();
		}

		std::tuple<std::string, Array2D<float>> convert(const std::string &data, int pos, int width) {
			Array2D<float> result = Array2D<float>(int2{ width, _nNeurons });
			std::string resultStr = "";
			const int dataLength = static_cast<int>(data.length());

			for (int i = -width; i < 0; ++i) {
				const int posInData = pos + i;
				const char c = ((posInData >= 0) && (posInData < dataLength)) ? data[posInData] : ' ';
				resultStr += c;				
				const std::vector<float> charData = getCharData(c);

				for (int j = 0; j < _nNeurons; ++j) {
					result.set(i+width, j, charData[j]);
				}
			}
			return std::tuple<std::string, Array2D<float>>(resultStr, result);
		}

		// return string with match per char
		std::tuple<std::string, std::vector<float>> convert(const Array2D<float> &data) {
			const int width = data._size.x;
			
			std::string strResult = "";
			std::vector<float> match(width);
			std::vector<float> charData(_nNeurons);

			for (int i = 0; i < width; ++i) {
				for (int j = 0; j < _nNeurons; ++j) {
					charData[j] = data.get(i, j);
				}
				const std::tuple<char, float> pred = getBestMatch(charData);
				strResult += std::get<0>(pred);
				match[i] = std::get<1>(pred);
			}
			return std::tuple<std::string, std::vector<float>>(strResult, match);
		}

		void addShotNoise(Array2D<float> &data, const float noisePercent) {
			std::mt19937 generator(0xDEADBEEF);
			std::uniform_real_distribution<> dist(0, 1);

			for (size_t i = 0; i < data._data_float.size(); ++i) {
				const float r = dist(generator);
				if (r < noisePercent) {
					data._data_float[i] = (data._data_float[i] == 1.0f) ? 0.0f : 1.0f;
				}
			}
		}

		static void test() {
			const std::string exampleX = "Dodgson told the girls a story that featured a bored little girl named Alice who goes looking for an adventure. The girls loved it, and Alice Liddell asked Dodgson to write it down for her. He began writing the manuscript of the story the next day, although that earliest version no longer exists. The girls and Dodgson took another boat trip a month later when he elaborated the plot to the story of Alice, and in November he began working on the manuscript in earnest.";



			TextConverter textConverter(0.04f, 200);
			const auto input = textConverter.convert(exampleX, 100, 40);
			Array2D<float> sdr = std::get<1>(input);
			plots::plotImage(sdr, 2, "text");

			const auto pred1 = textConverter.convert(sdr);
			const std::string predStr1 = std::get<0>(pred1);

			textConverter.addShotNoise(sdr, 0.1);
			const auto pred2 = textConverter.convert(sdr);
			const std::string predStr2 = std::get<0>(pred2);

			std::cout << "INFO: TextConverter:test: text1 = " << predStr1 << std::endl;
			std::cout << "INFO: TextConverter:test: text2 = " << predStr2 << std::endl;
			for (size_t i = 0; i < predStr1.length(); ++i) {
				std::cout << "INFO: TextConverter:test: char = " << predStr1[i] << ": " << std::get<1>(pred1)[i] << "; " << predStr2[i] << ": " << std::get<1>(pred2)[i] << std::endl;
			}
		}

	private:

		void initRandom() {

			std::mt19937 generator(0xDEADBEEF);

			for (char c = 'a'; c <= 'z'; ++c) {
				_data.emplace(c, createRandomChar(_sparsity, _nNeurons, generator));
			}
			for (char c = 'A'; c <= 'Z'; ++c) {
				_data.emplace(c, createRandomChar(_sparsity, _nNeurons, generator));
			}
			_data.emplace(' ', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('.', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace(',', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace(';', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace(':', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('?', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('!', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('(', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace(')', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('[', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace(']', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('-', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('"', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('*', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('_', createRandomChar(_sparsity, _nNeurons, generator));
			_data.emplace('\'', createRandomChar(_sparsity, _nNeurons, generator));
		}

		std::vector<float> createRandomChar(const float sparsity, const int nNeurons, std::mt19937 &generator) const {
			std::uniform_int_distribution<> dist(0, nNeurons-1);
			std::vector<float> result(nNeurons, 0.0f);

			int count = 0;
			while (count < _nNeuronsActive) {
				const int neuronId = dist(generator);
				if (result[neuronId] == 0.0f) {
					result[neuronId] = 1.0f;
					count++;
				}
			}
			return result;
		}

		const std::vector<float> &getCharData(const char c) {

			if (isprint(c)) {
				//const std::vector<float> &result;
				if (_data.find(c) != _data.end()) {
					return _data[c];
				}
				else {
					std::cout << "WARNING: TextLoader:getCharData: could not find data for char " << c << "." << std::endl;
					return _data[' '];
				}
			}
			else {
				return _data[' '];
			}
		}

		float getMatch(const std::vector<float> &data, const char c) {
			float result = 0;
			const std::vector<float> charData = getCharData(c);
			for (int i = 0; i < _nNeurons; ++i) {
				result += data[i] * charData[i];
			}
			return result;
		}

		std::tuple<char, float> getBestMatch(const std::vector<float> &data) {
			float match = -1;
			char c = ' ';
			for (auto it : _data) {
				const float tmpMatch = getMatch(data, it.first);
				if (tmpMatch > match) {
					match = tmpMatch;
					c = it.first;
				}
			}
			return std::tuple<char, float>(c, match/_nNeuronsActive);
		}

	};
}