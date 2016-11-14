#pragma once

#include <cassert>
#include <cmath>	// for std::round
#include <iostream>

namespace feynman {

	// FixPoint Parameters
	const bool useFixPoint = false;
	
	
	const bool UPDATE_FLOATING_POINT = true;
	const bool UPDATE_FIXED_POINT = true;

	const unsigned int N_BITS_FIXPOINT = 8;
	const unsigned int N_BITS_DENOMINATOR = 8;

	using FixPoint = unsigned __int8;
	using FixPoint2 = unsigned __int16;
	using FixPoint3 = unsigned __int32;

	// CODE

	const unsigned int maxValue = 1 << (N_BITS_FIXPOINT - N_BITS_DENOMINATOR);

	const unsigned __int64 DENOMINATOR = (1 << N_BITS_DENOMINATOR)-1;


	const unsigned __int64 DENOMINATOR_POW2 = DENOMINATOR * DENOMINATOR;
	const unsigned __int64 DENOMINATOR_POW3 = DENOMINATOR * DENOMINATOR * DENOMINATOR;

	bool inRange(const float f) {
		return (f >= 0.0) && (f <= maxValue);
	}

	FixPoint reducePower1(unsigned int i) {
		return static_cast<FixPoint>(i >> (N_BITS_DENOMINATOR * 1));
	}
	FixPoint reducePower2(unsigned long long i) {
		return static_cast<FixPoint>(i >> (N_BITS_DENOMINATOR * 2));
	}

	FixPoint addSaturate(FixPoint a, FixPoint b) {
		unsigned int tmp = static_cast<unsigned int>(a) + static_cast<unsigned int>(b);
		return (tmp > DENOMINATOR) ? DENOMINATOR : static_cast<FixPoint>(tmp);
	}

	FixPoint substractSaturate(FixPoint a, FixPoint b)
	{
		return (b > a) ? 0 : a - b;
	}


	float toFloat(const FixPoint fixPoint)
	{
		return static_cast<float>(static_cast<float>(fixPoint) / DENOMINATOR);
	}
	float toFloat(const FixPoint2 fixPoint)
	{
		assert(fixPoint <= DENOMINATOR);
		return static_cast<float>(static_cast<float>(fixPoint) / DENOMINATOR_POW2);
	}

	FixPoint toFixPoint(const float f) {
//#		ifdef _DEBUG
			if (!inRange(f)) std::printf("WARNING: toFixPoint: f=%f30.28 is not in range.\n", f);
//#		endif
		const FixPoint result = static_cast<FixPoint>(std::round(f * DENOMINATOR));
		if (false)
		{
			const float error = std::abs(f - toFloat(result));
			if (error > 0.002)
			{
				std::cout << "WARNING: toFixPoint: f=" << f << "; result=" << static_cast<unsigned long long>(result) << "; error=" << error << std::endl;
			}
		}
		return result;
	}
}
