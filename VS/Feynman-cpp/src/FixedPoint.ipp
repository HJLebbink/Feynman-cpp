#pragma once

#include <cassert>
#include <cmath>	// for std::round
#include <limits>	// for std::numeric_limits
#include <iostream>

namespace feynman {

	// FixPoint Parameters
//#define USE_FIXED_POINT
	
	
	const bool UPDATE_FLOATING_POINT = true;
	const bool UPDATE_FIXED_POINT = true;

	const unsigned int N_BITS_FIXPOINT = 8; // number of bits in the fixpoint
	const unsigned int N_BITS_DENOMINATOR = 7; // number of bits in the denominator of the fixpoint

	using FixedP =  __int8;
	using FixedP2 = __int16;
	using FixedP3 = __int32;

	// CODE

	const FixedP2 minValueFixedP = static_cast<FixedP2>(std::numeric_limits<FixedP>::min());
	const FixedP2 maxValueFixedP = static_cast<FixedP2>(std::numeric_limits<FixedP>::max());

	const float maxValueFloat = static_cast<float>(1llu << (N_BITS_FIXPOINT - N_BITS_DENOMINATOR))/2;
	const float minValueFloat = -maxValueFloat;
	const float stepSize = (maxValueFloat - minValueFloat) / N_BITS_FIXPOINT;

	const unsigned __int64 DENOMINATOR = (1 << N_BITS_DENOMINATOR)-1;
	const unsigned __int64 DENOMINATOR_POW2 = DENOMINATOR * DENOMINATOR;
	const unsigned __int64 DENOMINATOR_POW3 = DENOMINATOR * DENOMINATOR * DENOMINATOR;

	bool inRange(const float f) {
		return (f >= minValueFloat) && (f <= maxValueFloat);
	}

	FixedP reducePower1(FixedP2 i) {
		return static_cast<FixedP>(i >> (N_BITS_FIXPOINT * 1));
	}
	FixedP reducePower2(FixedP3 i) {
		return static_cast<FixedP>(i >> (N_BITS_FIXPOINT * 2));
	}

	FixedP saturate(const FixedP2 value) {
		return (value > maxValueFixedP)
			? (static_cast<FixedP>(maxValueFixedP)) :
			((value < minValueFixedP)
				? static_cast<FixedP>(minValueFixedP) :
				static_cast<FixedP>(value));
	}

	float saturateFloat(const float value) {
		return (value > maxValueFloat) ? maxValueFloat : ((value < minValueFloat) ? minValueFloat : value);
	}

	float toFloat(const FixedP fixPoint)
	{
		return static_cast<float>(static_cast<float>(fixPoint) / static_cast<int>(1 << (N_BITS_DENOMINATOR * 1)));
	}
	float toFloat(const FixedP2 fixPoint)
	{
		return static_cast<float>(static_cast<float>(fixPoint) / static_cast<int>(1 << (N_BITS_DENOMINATOR * 2)));
	}

	FixedP toFixedP(float value) {
		//#		ifdef _DEBUG
		if (!inRange(value)) {
			//printf("WARNING: Helpers::toFixedP: value (%24.22f) is not in range [%24.22f,%24.22f]; saturating.\n", value, minValueFloat, maxValueFloat);
			value = saturateFloat(value);
			//throw 1;
		}
		//#		endif

		return static_cast<FixedP>(std::round(value * (1 << (N_BITS_DENOMINATOR * 1))));
	}

	FixedP2 toFixedP2(float value) {
		//#		ifdef _DEBUG
		if (!inRange(value)) {
			//printf("WARNING: Helpers::toFixedP: value (%24.22f) is not in range [%24.22f,%24.22f]; saturating.\n", value, minValueFloat, maxValueFloat);
			value = saturateFloat(value);
		}
		//#		endif
		return static_cast<FixedP2>(std::round(value * (1 << (N_BITS_DENOMINATOR * 2))));
	}


	FixedP add_saturate(FixedP a, FixedP b)
	{
		//TODO convice the compiler to do saturation aritmethic
		return saturate(static_cast<FixedP2>(a) + static_cast<FixedP2>(b));
	}

	FixedP substract_saturate(FixedP a, FixedP b)
	{
		//TODO convice the compiler to do saturation aritmethic
		return saturate(static_cast<FixedP2>(a) - static_cast<FixedP2>(b));
	}

	FixedP2 multiply(FixedP a, FixedP b) {
		return static_cast<FixedP2>(a) * static_cast<FixedP2>(b);
	}

	FixedP multiply_saturate(const FixedP a, const FixedP b) {
		const FixedP2 a2 = static_cast<FixedP2>(a);
		const FixedP2 b2 = static_cast<FixedP2>(b);
		const FixedP2 ab2 = a2 * b2;
		const FixedP ab = ab2 >> (N_BITS_DENOMINATOR * 1);
		return ab;
	}

	FixedP multiply_saturate(FixedP2 a, FixedP b) {
		const FixedP3 a2 = static_cast<FixedP3>(a);
		const FixedP3 b2 = static_cast<FixedP3>(b);
		const FixedP3 ab2 = a2 * b2;
		const FixedP ab = ab2 >> (N_BITS_DENOMINATOR * 2);
		return ab;
	}

	namespace fixedPointTest {
		void test0() {
			{
				const float f = 0.0;
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest: f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
			{
				const float f = 0.5;
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest: f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
			{
				const float f = 1.0;
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest: f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
			{
				const float f = -0.5;
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest: f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
			{
				const float f = -1;
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest: f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
		}
		void test1() {
			const int nSteps = 100;
			const float delta = (maxValueFloat - minValueFloat) / nSteps;
			for (int i = 0; i < nSteps; ++i)
			{
				const float f = minValueFloat + (i * delta);
				const FixedP fp = toFixedP(f);
				std::printf("INFO: fixPointTest:test1 f=%24.22f; fixPoint=%i; toFloat(fp)=%24.22f; error=%24.22f\n", f, static_cast<int>(fp), toFloat(fp), (toFloat(fp) - f));
			}
		}
		void test2() {
			{
				const float f1 = 0.5f;
				const float f2 = 0.25f;
				const float f3 = f1 * f2;
				const FixedP fp1 = toFixedP(f1);
				const FixedP fp2 = toFixedP(f2);
				const FixedP fp3 = multiply_saturate(fp1, fp2);
				std::printf("INFO: fixPointTest:test2: f1(=%7.5f; %i) * f2(=%7.5f; %i) = f3(=%7.5f; %i); f1(=%7.5f) * f2(=%7.5f) = f3(=%7.5f)\n", f1, fp1, f2, fp2, f3, fp3, toFloat(fp1), toFloat(fp2), toFloat(fp3));
			}
			{
				const float f1 = 0.5f;
				const float f2 = 0.25f;
				const float f3 = f1 * f2;
				const FixedP2 fp1 = toFixedP2(f1);
				const FixedP fp2 = toFixedP(f2);
				const FixedP fp3 = multiply_saturate(fp1, fp2);
				std::printf("INFO: fixPointTest:test2: f1(=%7.5f; %i) * f2(=%7.5f; %i) = f3(=%7.5f; %i); f1(=%7.5f) * f2(=%7.5f) = f3(=%7.5f)\n", f1, fp1, f2, fp2, f3, fp3, toFloat(fp1), toFloat(fp2), toFloat(fp3));
			}
		}
	}
}
