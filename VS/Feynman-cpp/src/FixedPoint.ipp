#pragma once

#include <cassert>
#include <cmath>	// for std::round
#include <iostream>

namespace feynman {

	using FixPoint = unsigned __int8;
	const unsigned __int64 DENOMINATOR = 0xFF;
	const unsigned __int64 DENOMINATOR_POW2 = DENOMINATOR * DENOMINATOR;
	const unsigned __int64 DENOMINATOR_POW3 = DENOMINATOR * DENOMINATOR * DENOMINATOR;

	float toFloat(const FixPoint fixPoint)
	{
		assert(fixPoint <= DENOMINATOR, "toFloat: provided fixPoint be smaller or equal to DENOMINATOR");
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
