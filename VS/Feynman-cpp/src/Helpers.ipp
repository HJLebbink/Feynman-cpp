// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <array>
#include <vector>
#include <random>

namespace feynman {

	struct int2 { int x, y; };
	struct int3 { int x, y, z; };
	struct uint2 { unsigned int x, y; };
	struct float2 { float x, y; };

	template <class T>
	struct Image2D {
		std::vector<T> _data;
		int2 _size;

		// default constructor
		Image2D() : _size({ 0, 0 }) {}

		Image2D(int2 size) : _size(size)
		{
			int nElements = size.x * size.y;
			_data.resize(nElements);
		}
		void swap(Image2D& other) {
			_data.swap(other._data);
			int2 tmp = this->_size;
			this->_size = other._size;
			other._size = tmp;
		}
	};

	template <class T>
	struct Image3D {
		std::vector<T> _data;
		int3 _size;

		Image3D(int3 size) : _size(size)
		{
			int nElements = size.x * size.y * size.z;
			_data.resize(nElements);
		}
		void swap(Image3D& other) {
			_data.swap(other._data);
			int3 tmp = this->_size;
			this->_size = other._size;
			other._size = tmp;
		}
	};

	int inline pos(const int2 coord, const int2 size) {
		return (coord.x * size.y) + coord.y;
	}

	int inline pos(const int coord_x, const int coord_y, const int size_y) {
		return (coord_x * size_y) + coord_y;
	}

	int inline pos(const int3 coord, const int3 size) {
		return (coord.x * size.y * size.z) + (coord.y * size.z) + coord.z;
	}

	int inline pos(const int coord_x, const int coord_y, const int coord_z, const int size_y, const int size_z) {
		return (coord_x * size_y * size_z) + (coord_y * size_z) + coord_z;
	}


	template <class T>
	T inline read_2D(const Image2D<T> &image, const int2 coord)
	{
		return image._data[pos(coord, image._size)];
	}

	template <class T>
	T inline read_2D(const Image2D<T> &image, const int coord_x, const int coord_y)
	{
		const int idx = pos(coord_x, coord_y, image._size.y);
		//std::cout << "read_imagef_2D: idx=" << idx << "; x=" << coord_x << "; y=" << coord_y << std::endl;
		return image._data[idx];
	}

	template <class T>
	T inline read_3D(const Image3D<T> &image, const int coord_x, const int coord_y, const int coord_z) {
		const int idx = pos(coord_x, coord_y, coord_z, image._size.y, image._size.z);
		//std::cout << "read_imagef_3D: idx=" << idx << "; x=" << coord_x << "; y=" << coord_y << "; z=" << coord_z << std::endl;
		return image._data[idx];
	}

	template <class T>
	void inline write_2D(Image2D<T> &image, const int coord_x, const int coord_y, const T v) {
		image._data[pos(coord_x, coord_y, image._size.y)] = v;
	}

	template <class T>
	void inline write_3D(Image3D<T> &image, const int coord_x, const int coord_y, const int coord_z, const T v) {
		image._data[pos(coord_x, coord_y, coord_z, image._size.y, image._size.z)] = v;
	}

	float randFloat(uint2* state) {
		const float invMaxInt = 1.0f / 4294967296.0f;
		unsigned int x = (*state).x * 17 + (*state).y * 13123;
		(*state).x = (x << 13) ^ x;
		(*state).y ^= (x << 7);
		unsigned int tmp = x * (x * x * 15731 + 74323) + 871483;
		return static_cast<float>(tmp) * invMaxInt;
	}

	//Buffer types (can be used as indices)
	enum BufferType {
		_front = 0, _back = 1
	};

	//Double buffer types
	template <class T>
	using DoubleBuffer2D = std::vector<Image2D<T>>;
	template <class T>
	using DoubleBuffer3D = std::vector<Image3D<T>>;

	//Double buffer initialization helpers
	template <class T>
	DoubleBuffer2D<T> createDoubleBuffer2D(int2 size) {
		return { Image2D<T>(size), Image2D<T>(size) };
	}
	template <class T>
	DoubleBuffer3D<T> createDoubleBuffer3D(int3 size) {
		return { Image3D<T>(size), Image3D<T>(size) };
	}

	static void clear(Image2D<float> &image) {
		const size_t nBytes = image._size.x * image._size.y * 4;
		memset(&image._data[0], 0, nBytes);
	}
//	static void clear(Image2D<float2> &image) {
//		fill(image, { 0.0f, 0.0f });
//	}

	static void copy(const Image2D<float> &src, Image2D<float> &dst) {
		const size_t nBytes = src._size.x * src._size.y * 4;
		memcpy(&dst._data[0], &src._data[0], nBytes);
	}
	static void copy(const Image2D<float> &src, std::vector<float> &dst) {
		const size_t nBytes = src._size.x * src._size.y * 4;
		memcpy(&dst[0], &src._data[0], nBytes);
	}
	static void copy(std::vector<float> &src, Image2D<float> &dst) {
		const size_t nBytes = dst._size.x * dst._size.y * 4;
		memcpy(&dst._data[0], &src[0], nBytes);
	}


	// Initialize a random uniform 2D image (X field)
	template <class T>
	void randomUniform2D(
		Image2D<T> &image2D, 
		const int2 range, 
		const float2 minMax,
		std::mt19937 &rng) 
	{
		std::uniform_int_distribution<int> seedDist(0, 999);
		const unsigned int seedx = static_cast<unsigned int>(seedDist(rng));
		const unsigned int seedy = static_cast<unsigned int>(seedDist(rng));;

		for (int x = 0; x < range.x; ++x) {
			for (int y = 0; y < range.y; ++y) {

				uint2 seedValue;
				seedValue.x = seedx + ((x * 29) + 12) * 36;
				seedValue.y = seedy + ((y * 16) + 23) * 36;
				const float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;
				write_2D(image2D, x, y, value);
			}
		}
	}

	// Initialize a random uniform 3D image (X field)
	template <class T>
	void randomUniform3D(
		Image3D<T> &values,
		const int3 range,
		const float2 minMax,
		std::mt19937 &rng)
	{
		std::uniform_int_distribution<int> seedDist(0, 999);
		const unsigned int seedx = static_cast<unsigned int>(seedDist(rng));
		const unsigned int seedy = static_cast<unsigned int>(seedDist(rng));;

		for (int x = 0; x < range.x; ++x) {
			for (int y = 0; y < range.y; ++y) {
				for (int z = 0; z < range.z; ++z) {
					uint2 seedValue;
					seedValue.x = seedx + (((x * 12) + 76 + z) * 3) * 12;
					seedValue.y = seedy + (((x * 21) + 42 + z) * 7) * 12;
					const float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;
					write_3D(values, x, y, z, value);
				}
			}
		}
	}

	float randNormal(uint2* state) {
		float u1 = randFloat(state);
		float u2 = randFloat(state);
		return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
	}

	float sigmoid(float x) {
		return 1.0f / (1.0f + exp(-x));
	}

	float relu(float x, float leak) {
		x += 0.5f;

		if (x > 1.0f)
			return 1.0f + (x - 1.0f) * leak;

		return x > 0.0f ? x : x * leak;
	}

	float relud(float x, float leak) {
		x += 0.5f;
		return x > 0.0f && x < 1.0f ? 1.0f : leak;
	}
	/*
	inline bool inBounds0(int2 position, int2 upperBound) {
		return (position.x >= 0) && (position.x < upperBound.x) && (position.y >= 0) && (position.y < upperBound.y);
	}
	*/
	inline bool inBounds0(int pos_x, int pos_y, int max_x, int max_y) {
		return (pos_x >= 0) && (pos_x < max_x) && (pos_y >= 0) && (pos_y < max_y);
	}

	inline bool inBounds0(const int pos, const int max) {
		return (pos >= 0) && (pos < max);
	}

	bool inBounds(int2 position, int2 lowerBound, int2 upperBound) {
		return (position.x >= lowerBound.x) && (position.x < upperBound.x) && (position.y >= lowerBound.y) && (position.y < upperBound.y);
	}

	bool inBounds(const int position, const int lowerBound, const int upperBound) {
		return (position >= lowerBound) && (position < upperBound);
	}



	inline int2 project(int2 position, float2 toScalars) {
		int2 r;
		r.x = static_cast<int>((position.x * toScalars.x) + 0.5f);
		r.y = static_cast<int>((position.y * toScalars.y) + 0.5f);
		return r;
	}

	inline int project(const int position, const float toScalars) {
		return static_cast<int>((position * toScalars) + 0.5f);
	}

	inline float2 find_min_max(const Image2D<float> &image) {
		float minimum = 1000000;
		float maximum = -1000000;
		for (int i = 0; i < image._size.x * image._size.y; ++i) {
			minimum = std::min(minimum, image._data[i]);
			maximum = std::max(maximum, image._data[i]);
		}
		return{ minimum, maximum };
	}
}