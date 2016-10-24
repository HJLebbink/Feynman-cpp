#pragma once


#ifdef _MSC_VER		// compiler: Microsoft Visual Studio
	#include <windows.h>
#else
	#include <x86intrin.h>
#endif
#define rdtsc __rdtsc


namespace tools {

	static unsigned long long timing_start, timing_end;

	static inline void reset_and_start_timer() {
		timing_start = rdtsc();
	}

	// Returns the number of millions of elapsed processor cycles since the last reset_and_start_timer() call.
	static inline double get_elapsed_mcycles() {
		timing_end = rdtsc();
		return (timing_end - timing_start) / (1024. * 1024.);
	}

	// Returns the number of thousands of elapsed processor cycles since the last reset_and_start_timer() call.
	static inline double get_elapsed_kcycles()
	{
		timing_end = rdtsc();
		return (timing_end - timing_start) / (1024.);
	}

	// Returns the number of elapsed processor cycles since the last reset_and_start_timer() call.
	static inline unsigned long long get_elapsed_cycles()
	{
		timing_end = rdtsc();
		return (timing_end - timing_start);
	}
}
