// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <random>
#include <ctime>
#include "FixedPoint.ipp"

namespace feynman {

	//Possible encoder identifiers
	 enum SparseFeaturesType {
		_stdp, _delay, _chunk
	};

	std::mt19937 generator_test(static_cast<unsigned int>(time(nullptr)));


	struct int2 { int x, y; };
	struct int3 { int x, y, z; };
	struct uint2 { unsigned int x, y; };
	struct float2 { float x, y; };

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

	template <typename T>
	struct Array2D {
		std::vector<T> _data_float;
#		ifdef USE_FIXED_POINT
		std::vector<FixedP> _data_fixP;
#		endif
		int2 _size;


		Array2D(int2 size) : _size(size)
		{
			const int nElements = size.x * size.y;
			_data_float.resize(nElements);
#			ifdef USE_FIXED_POINT
			_data_fixP.resize(nElements);
#			endif
		}

		// default constructor
		Array2D() : Array2D(int2{ 0, 0 }) {}
		Array2D(int x, int y) : Array2D(int2{ x, y }) {}
	
		void fill(T value) {
			const int nElements = _size.x * _size.y;
			for (int i = 0; i < nElements; ++i) {
				_data_float[i] = value;
			}
		}

		void set(int x, int y, T value) {
			_data_float[pos(x, y, _size.y)] = value;
		}

		float get(int x, int y) const {
			return _data_float[pos(x, y, _size.y)];
		}
	
		void swap(Array2D<T>& other) {
			_data_float.swap(other._data_float);
#			ifdef USE_FIXED_POINT
			_data_fixP.swap(other._data_fixP);
#			endif
			const int2 tmp = this->_size;
			this->_size = other._size;
			other._size = tmp;
		}

		int2 getSize() const {
			return _size;
		}
	};

	struct Image3D {
		std::vector<float> _data_float;
#		ifdef USE_FIXED_POINT
		std::vector<FixedP> _data_fixP;
#		endif
		int3 _size;

		Image3D(int3 size) : _size(size)
		{
			const int nElements = size.x * size.y * size.z;
			_data_float.resize(nElements);
#			ifdef USE_FIXED_POINT
			_data_fixP.resize(nElements);
#			endif
		}

		// default constructor
		Image3D() : Image3D(int3{ 0, 0, 0 }) {}


		void swap(Image3D& other) {
			_data_float.swap(other._data_float);
#			ifdef USE_FIXED_POINT
			_data_fixP.swap(other._data_fixP);
#			endif
			const int3 tmp = this->_size;
			this->_size = other._size;
			other._size = tmp;
		}

		int3 getSize() const {
			return _size;
		}
	};


	//TODO rename Image2D to Array2Df
	using Image2D = Array2D<float>;
	using Array2Di2 = Array2D<int2>;
	using Array2D2f = Array2D<float2>;

	template <typename T>
	T inline read_2D(const Array2D<T> &image, const int coord_x, const int coord_y)
	{
		const int idx = pos(coord_x, coord_y, image._size.y);
		//std::cout << "read_imagef_2D: idx=" << idx << "; x=" << coord_x << "; y=" << coord_y << std::endl;
		const T value = image._data_float[idx];
#		ifdef _DEBUG
//		if (false) {
//			if (!inRange(value)) {
//				std::cout << "WARNING: Helpers::read_2D: value "<< value <<" is not in range ["<< minValueFloat <<","<< maxValueFloat <<"]" << std::endl;
//				//throw 1;
//			}
//		}
#		endif
#		ifdef USE_FIXED_POINT
		if (false) {
			float value_fp = toFloat(image._data_fixP[idx]);
			if (std::abs(value_fp - value) > 0.004)
			{
				//printf("WARNING: Helpers::read_2D: value_fp=%30.28f; value=%30.28f\n", value_fp, value);
				//throw 1;
			}
		}
#		endif
		return value;
	}

	template <typename T>
	float inline read_2D(const Array2D<T> &image, const int2 coord)
	{
		return read_2D(image, coord.x, coord.y);
	}

	FixedP inline read_2D_fixp(const Image2D &image, const int coord_x, const int coord_y)
	{
		const int idx = pos(coord_x, coord_y, image._size.y);
#		ifdef USE_FIXED_POINT
		FixedP value = image._data_fixP[idx];
#		else
		FixedP value = toFixedP(image._data_float[idx]);
#		ifdef _DEBUG
		if (false) {
			const FixedP value2 = toFixedP(image._data_float[idx]);
			if (value != value2) {
				//printf("WARNING: read_2D_fixp: idx=%i; x=%i; y=%i; value=%llu; value2=%llu\n", idx, coord_x, coord_y, static_cast<unsigned long long>(value), static_cast<unsigned long long>(value2));
				//throw 1;
			}
		}
		#endif
#		endif
		return value;
	}

	float inline read_3D(const Image3D &image, const int coord_x, const int coord_y, const int coord_z) {
		const int idx = pos(coord_x, coord_y, coord_z, image._size.y, image._size.z);
		//std::cout << "read_3D: idx=" << idx << "; x=" << coord_x << "; y=" << coord_y << "; z=" << coord_z << std::endl;
		const float value = image._data_float[idx];
#		ifdef _DEBUG
		if (false) {
			if (!inRange(value)) {
				//printf("WARNING: Helpers::read_3D: value (%24.22f) is not in range [%24.22f,%24.22f]\n", value, minValueFloat, maxValueFloat);
				//throw 1;
			}
		}
#		endif
		return value;
	}

	FixedP inline read_3D_fixp(const Image3D &image, const int coord_x, const int coord_y, const int coord_z) {
		const int idx = pos(coord_x, coord_y, coord_z, image._size.y, image._size.z);
#		ifdef USE_FIXED_POINT
		FixedP value = image._data_fixP[idx];
#		else
		FixedP value = toFixedP(image._data_float[idx]);
#		ifdef _DEBUG
		if (false) {
			const FixedP value2 = toFixedP(image._data_float[idx]);
			if (value != value2) {
				std::cout << "WARNING: read_3D_fixp: idx=" << idx << "; x=" << coord_x << "; y=" << coord_y << "; value=" << static_cast<unsigned long long>(value) << "; value2=" << static_cast<unsigned long long>(value2) << std::endl;
				throw 1;
			}
		}
#		endif
#		endif
		return value;
	}

	template <typename T>
	void inline write_2D(Array2D<T> &image, const int coord_x, const int coord_y, const T value) {

		image._data_float[pos(coord_x, coord_y, image._size.y)] = value;
#		ifdef USE_FIXED_POINT
		image._data_fixP[pos(coord_x, coord_y, image._size.y)] = toFixedP(value);
#		endif
	}

	void inline write_2D_fixp(Image2D &image, const int coord_x, const int coord_y, const FixedP value) {
#		ifdef USE_FIXED_POINT
		image._data_fixP[pos(coord_x, coord_y, image._size.y)] = value;
#		endif
		image._data_float[pos(coord_x, coord_y, image._size.y)] = toFloat(value);
	}

	void inline write_3D(Image3D &image, const int coord_x, const int coord_y, const int coord_z, float value) {
		image._data_float[pos(coord_x, coord_y, coord_z, image._size.y, image._size.z)] = value;
#		ifdef USE_FIXED_POINT
		image._data_fixP[pos(coord_x, coord_y, coord_z, image._size.y, image._size.z)] = toFixedP(value);
#		endif
	}

	void inline write_3D_fixp(Image3D &image, const int coord_x, const int coord_y, const int coord_z, const FixedP value) {
#		ifdef USE_FIXED_POINT
		image._data_fixP[pos(coord_x, coord_y, coord_z, image._size.y, image._size.z)] = value;
#		endif
		image._data_float[pos(coord_x, coord_y, coord_z, image._size.y, image._size.z)] = toFloat(value);
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
	using DoubleBuffer2D = std::vector<Image2D>;
	using DoubleBuffer3D = std::vector<Image3D>;

	using DoubleBuffer2D2f = std::vector<Array2D2f>;


	//Double buffer initialization helpers
	DoubleBuffer2D createDoubleBuffer2D(int2 size) {
		return { Image2D(size), Image2D(size) };
	}
	DoubleBuffer2D2f createDoubleBuffer2D2f(int2 size) {
		return{ Array2D2f(size), Array2D2f(size) };
	}

	DoubleBuffer3D createDoubleBuffer3D(int3 size) {
		return { Image3D(size), Image3D(size) };
	}

	static void clear(Image2D &image) {
		const size_t nBytes = image._size.x * image._size.y * sizeof(float);
		memset(&image._data_float[0], 0, nBytes);
	}
	static void clear(Array2D2f &image) {
		const size_t nBytes = image._size.x * image._size.y * sizeof(float) * 2;
		memset(&image._data_float[0], 0, nBytes);
	}

	static void clear(Image3D &image) {
		const size_t nBytes = image._size.x * image._size.y * image._size.z * sizeof(float);
		memset(&image._data_float[0], 0, nBytes);
	}

	static void copy(const Image2D &src, Image2D &dst) {
		const size_t nBytes = src._size.x * src._size.y * sizeof(float);
		memcpy(&dst._data_float[0], &src._data_float[0], nBytes);
#		ifdef USE_FIXED_POINT
		const size_t nBytesFixP = src._size.x * src._size.y * sizeof(FixedP);
		memcpy(&dst._data_fixP[0], &src._data_fixP[0], nBytesFixP);
#		endif
	}
	static void copy(const Image2D &src, std::vector<float> &dst) {
		const size_t nBytes = src._size.x * src._size.y * sizeof(float);
		memcpy(&dst[0], &src._data_float[0], nBytes);
	}
	static void copy(const std::vector<float> &src, Image2D &dst) {
		const size_t nBytes = dst._size.x * dst._size.y * sizeof(float);
		memcpy(&dst._data_float[0], &src[0], nBytes);
#		ifdef USE_FIXED_POINT
		for (int i = 0; i < dst._size.x * dst._size.y; ++i) {
			dst._data_fixP[i] = toFixedP(src[i]);
		}
#		endif
	}
	static void copy(const Image3D &src, Image3D &dst) {
		const size_t nBytes = src._size.x * src._size.y * src._size.z * sizeof(float);
		memcpy(&dst._data_float[0], &src._data_float[0], nBytes);
#		ifdef USE_FIXED_POINT
		const size_t nBytesFixP = src._size.x * src._size.y * src._size.z * sizeof(FixedP);
		memcpy(&dst._data_fixP[0], &src._data_fixP[0], nBytesFixP);
#		endif
	}

	// Initialize a random uniform 2D image (X field)
	void randomUniform2D(
		Image2D &image2D, 
		const int2 range, 
		const float2 minMax,
		std::mt19937 &rng) 
	{
		//TODO remove range

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
	void randomUniform3D(
		Image3D &values,
		const int3 range,
		const float2 minMax,
		std::mt19937 &rng)
	{
		std::uniform_int_distribution<int> seedDist(0, 999);
		const unsigned int seedx = static_cast<unsigned int>(seedDist(rng));
		const unsigned int seedy = static_cast<unsigned int>(seedDist(rng));;

		for (int x = 0; x < range.x; ++x) {
			for (int y = 0; y < range.y; ++y) {
#				pragma ivdep
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

	inline bool inBounds(int pos_x, int pos_y, int max_x, int max_y) {
		return (pos_x >= 0) && (pos_x < max_x) && (pos_y >= 0) && (pos_y < max_y);
	}

	inline bool inBounds(const int pos, const int max) {
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

	static std::tuple<int2, int2> cornerCaseRange(
		const int2 hiddenRange,
		const int2 visibleRange,
		const int radius,
		const float2 hiddenToVisible)
	{
		int2 rangeX = int2{ hiddenRange.x, hiddenRange.x };

		for (int x = 0; x < hiddenRange.x; ++x) {
			const int visiblePositionCenter_x = project(x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - radius;
			if (inBounds(fieldLowerBound_x, visibleRange.x)) {
				rangeX.x = x;
				break;
			}
		}
		for (int x = rangeX.x + 1; x < hiddenRange.x; ++x) {
			const int visiblePositionCenter_x = project(x, hiddenToVisible.x);
			const int fieldUpperBound_x = visiblePositionCenter_x + radius;
			if (!inBounds(fieldUpperBound_x, visibleRange.x)) {
				rangeX.y = x;
				break;
			}
		}
		return{ rangeX, rangeX };
	}
}