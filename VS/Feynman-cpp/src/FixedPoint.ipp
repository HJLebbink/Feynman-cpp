#pragma once

#include <cassert>
#include <cmath>	// for std::round
#include <limits>	// for std::numeric_limits
#include <iostream>

namespace feynman {

	// FixPoint Parameters
	const bool useFixPoint = false;
	
	
	const bool UPDATE_FLOATING_POINT = true;
	const bool UPDATE_FIXED_POINT = true;

	const unsigned int N_BITS_FIXPOINT = 8; // number of bits in the fixpoint
	const unsigned int N_BITS_DENOMINATOR = 8; // number of bits in the denominator of the fixpoint

	using FixPoint = __int8;
	using FixPoint2 = __int16;
	using FixPoint3 = __int32;

	// CODE

	const FixPoint2 maxValueFixPoint = static_cast<FixPoint2>(std::numeric_limits<FixPoint>::max());
	const FixPoint2 minValueFixPoint = static_cast<FixPoint2>(std::numeric_limits<FixPoint>::min());

	const float maxValueFloat = static_cast<float>((1 << (N_BITS_FIXPOINT - N_BITS_DENOMINATOR))/2);
	const float minValueFloat = -maxValueFloat;
	const float stepSize = (maxValueFloat - minValueFloat) / N_BITS_FIXPOINT;

	const unsigned __int64 DENOMINATOR = (1 << N_BITS_DENOMINATOR)-1;
	const unsigned __int64 DENOMINATOR_POW2 = DENOMINATOR * DENOMINATOR;
	const unsigned __int64 DENOMINATOR_POW3 = DENOMINATOR * DENOMINATOR * DENOMINATOR;

	bool inRange(const float f) {
		return (f >= -maxValueFloat) && (f <= maxValueFloat);
	}

	FixPoint reducePower1(unsigned int i) {
		return static_cast<FixPoint>(i >> (N_BITS_DENOMINATOR * 1));
	}
	FixPoint reducePower2(unsigned long long i) {
		return static_cast<FixPoint>(i >> (N_BITS_DENOMINATOR * 2));
	}

	FixPoint addSaturate(FixPoint a, FixPoint b) {
		const FixPoint2 tmp = static_cast<FixPoint2>(a) + static_cast<FixPoint2>(b);
		return (tmp > maxValueFixPoint) ? static_cast<FixPoint>(maxValueFixPoint) : static_cast<FixPoint>(tmp);
	}

	FixPoint substractSaturate(FixPoint a, FixPoint b)
	{
		const FixPoint2 tmp = static_cast<FixPoint2>(a) - static_cast<FixPoint2>(b);
		return (tmp > maxValueFixPoint) ? static_cast<FixPoint>(minValueFixPoint) : static_cast<FixPoint>(tmp);
	}

	float toFloat(const FixPoint fixPoint)
	{
		return static_cast<float>(static_cast<float>(fixPoint) / DENOMINATOR) - maxValueFloat;
	}
	float toFloat(const FixPoint2 fixPoint)
	{
		assert(fixPoint <= DENOMINATOR);
		return static_cast<float>(static_cast<float>(fixPoint) / DENOMINATOR_POW2) - maxValueFloat;
	}

	FixPoint toFixPoint(const float f) {
//#		ifdef _DEBUG
			if (!inRange(f)) std::printf("WARNING: toFixPoint: f=%30.28f is not in range %30.28f %30.28f.\n", f, -maxValue, maxValue);
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
