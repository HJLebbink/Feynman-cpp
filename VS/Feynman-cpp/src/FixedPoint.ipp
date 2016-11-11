#pragma once

#include <cassert>
#include <cmath>	// for std::round
#include <iostream>

namespace feynman {


	using FixPoint = unsigned __int8;
	const unsigned __int64 DENOMINATOR = 0xFF;
	const unsigned int nBits = 8;



	const unsigned __int64 DENOMINATOR_POW2 = DENOMINATOR * DENOMINATOR;
	const unsigned __int64 DENOMINATOR_POW3 = DENOMINATOR * DENOMINATOR * DENOMINATOR;

	FixPoint reducePower1(unsigned int i) {
		return static_cast<FixPoint>(i >> (nBits * 1));
	}
	FixPoint reducePower2(unsigned int i) {
		return static_cast<FixPoint>(i >> (nBits * 2));
	}

	FixPoint addSaturate(FixPoint a, FixPoint b) {
		unsigned int tmp = static_cast<unsigned int>(a) + static_cast<unsigned int>(b);
		if (tmp & )
	}


	float toFloat(const FixPoint fixPoint)
	{
		assert(fixPoint <= DENOMINATOR);
		return static_cast<float>(static_cast<double>(fixPoint) / DENOMINATOR);
	}

	FixPoint toFixPoint(const float f)
	{
		const FixPoint result = static_cast<FixPoint>(std::round(f * DENOMINATOR));

//#		ifdef _DEBUG
		if (f < 0.0f) std::cout << "WARNING: toFixPoint: f=" << f << " is smaller than 0." << std::endl;
		if (f > 1.0f) std::cout << "WARNING: toFixPoint: f=" << f << " is larger than 1." << std::endl;

		if (false)
		{
			const float error = std::abs(f - toFloat(result));
			if (error > 0.002)
			{
				std::cout << "WARNING: toFixPoint: f=" << f << "; result=" << static_cast<unsigned long long>(result) << "; error=" << error << std::endl;
			}
		}
//#		endif
		return result;
	}


}
