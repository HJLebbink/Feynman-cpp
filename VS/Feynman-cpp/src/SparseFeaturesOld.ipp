// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>		// for cerr and cout
#include <tuple>
#include <vector>
#include <random>
#include <algorithm> // for std::min
#include <ctime>

#include "timing.h"
#include "Helpers.ipp"
#include "SparseFeatures.ipp"

namespace feynman {

	//Sparse predictor
	//Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
	class SparseFeaturesOld: public SparseFeatures {
	public:

		//Visible layer desc
		struct VisibleLayerDesc {
			//Size of layer
			int2 _size;

			//Radius onto input
			int _radius;

			//Whether or not the middle (center) input should be ignored (self in recurrent schemes)
			bool _ignoreMiddle;

			//Learning rate
			float _weightAlpha;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(6), _ignoreMiddle(false),
				_weightAlpha(0.004f)
			{}
		};

		//Visible layer
		struct VisibleLayer {
			//Possibly manipulated input
			DoubleBuffer2D<float> _derivedInput;

			DoubleBuffer3D<float> _weights;
			float2 _hiddenToVisible;
			float2 _visibleToHidden;
			int2 _reverseRadii;
		};

		//Sparse Features Old Descriptor
		class SparseFeaturesOldDesc : public SparseFeatures::SparseFeaturesDesc {
		public:
			std::vector<VisibleLayerDesc> _visibleLayerDescs;
			int2 _hiddenSize;
			int _inhibitionRadius;
			float _activeRatio;
			float _biasAlpha;
			float2 _initWeightRange;
			std::mt19937 _rng;

			//Defaults
			SparseFeaturesOldDesc()
				: _hiddenSize({ 16, 16 }),
				_inhibitionRadius(8),
				_activeRatio(0.02f),
				_biasAlpha(0.01f),
				_initWeightRange({ 0.0f, 1.0f }),
				_rng()
			{
				_name = "old";
			}

			size_t getNumVisibleLayers() const override {
				return _visibleLayerDescs.size();
			}

			int2 getVisibleLayerSize(int vli) const override {
				return _visibleLayerDescs[vli]._size;
			}

			int2 getHiddenSize() const override {
				return _hiddenSize;
			}

			//Factory
			std::shared_ptr<SparseFeatures> sparseFeaturesFactory() override {
				return std::make_shared<SparseFeaturesOld>(_visibleLayerDescs, _hiddenSize, _inhibitionRadius, _activeRatio, _biasAlpha, _initWeightRange, _rng);
			}
		};

	private:

		// Hidden activations, states, biases, errors, predictions
		DoubleBuffer2D<float> _hiddenActivations;
		DoubleBuffer2D<float> _hiddenStates;
		DoubleBuffer2D<float> _hiddenBiases;

		//Hidden size
		int2 _hiddenSize;

		//Lateral inhibitory radius
		int _inhibitionRadius;
		
		float _activeRatio;
		float _biasAlpha;

		//Hidden summation temporary buffer
		DoubleBuffer2D<float> _hiddenSummationTemp;

		//Layers and descs
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;
		
	public:
	
		//Default constructor
		SparseFeaturesOld() {};

		/*!
		\brief Create a comparison sparse coder with random initialization.
		Requires initialization information.
		\param visibleLayerDescs descriptors for each input layer.
		\param hiddenSize hidden layer (SDR) size (2D).
		\param inhibitionRadius inhibitory radius.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		SparseFeaturesOld(
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const int2 hiddenSize,
			const int inhibitionRadius,
			const float activeRatio,
			const float biasAlpha,
			const float2 initWeightRange,
			std::mt19937 &rng
		) :
			_hiddenSize(hiddenSize),
			_inhibitionRadius(inhibitionRadius),
			_activeRatio(activeRatio),
			_biasAlpha(biasAlpha)
		{
			_type = SparseFeaturesType::_old;
			
			_visibleLayerDescs = visibleLayerDescs;
			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				vl._hiddenToVisible = float2{ 
					static_cast<float>(vld._size.x) / _hiddenSize.x,
					static_cast<float>(vld._size.y) / _hiddenSize.y
				};
				vl._visibleToHidden = float2{ 
					static_cast<float>(_hiddenSize.x) / vld._size.x,
					static_cast<float>(_hiddenSize.y) / vld._size.y
				};
				vl._reverseRadii = int2{ 
					static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
				};
				{
					const int weightDiam = vld._radius * 2 + 1;
					const int numWeights = weightDiam * weightDiam;
					const int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._weights = createDoubleBuffer3D<float>(weightsSize);
					randomUniform3D(vl._weights[_back], initWeightRange, rng);
				}

				vl._derivedInput = createDoubleBuffer2D<float>(vld._size);
				clear(vl._derivedInput[_back]);
			}

			// Hidden state data
			_hiddenActivations = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenStates = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenBiases = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenSummationTemp = createDoubleBuffer2D<float>(_hiddenSize);

			clear(_hiddenActivations[_back]);
			clear(_hiddenStates[_back]);

			if (true) {
				randomUniform2D(_hiddenBiases[_back], initWeightRange, rng);
			} else {
				clear(_hiddenBiases[_back]);
			}
		}

		/*!
		\brief Activate predictor
		\param visibleStates the input layer states.
		\param activeRatio % active units.
		\param rng a random number generator.
		*/
		void activate(
			const std::vector<Array2D<float>> &visibleStates,
			const Array2D<float> &predictionsPrev,
			std::mt19937 &rng) override
		{
			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);
			//plots::plotImage(_hiddenSummationTemp[_back], 6, "SFOld:activate:hiddenSummationTemp-1");

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				plots::plotImage(visibleStates[vli], 6, "SFOld:activate:visibleStates" + std::to_string(vli));

				spDeriveInputs(
					visibleStates[vli],			// in
					//vl._derivedInput[_back],	// unused
					vl._derivedInput[_front],	// out
					vld._size
				);
				//plots::plotImage(vl._derivedInput[_front], 6, "SFOld:activate:derivedInput" + std::to_string(vli));

				spStimulus(
					vl._derivedInput[_front],		// in
					_hiddenSummationTemp[_back],	// in
					_hiddenSummationTemp[_front],	// out
					vl._weights[_back],				// in
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					vld._ignoreMiddle
				);
				plots::plotImage(_hiddenSummationTemp[_front], 6, "SFOld:activate:hiddenSummationTemp" + std::to_string(vli));

				// Swap buffers
				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}

			// Activate
			spActivate(
				_hiddenSummationTemp[_back],	// in
				//_hiddenStates[_back],			// unused
				_hiddenBiases[_back],			// in
				//_hiddenActivations[_back],	// unused
				_hiddenActivations[_front],		// out
				_hiddenSize
			);
			//plots::plotImage(_hiddenSummationTemp[_back], 6, "SFOld:activate:hiddenSummationTemp");
			//plots::plotImage(_hiddenActivations[_front], 6, "SFOld:activate:hiddenActivations");

			// Inhibit
			spInhibit(
				_hiddenActivations[_front],		// in
				_hiddenStates[_front],			// out
				_hiddenSize,
				_inhibitionRadius,
				_activeRatio,
				_hiddenSize
			);
			plots::plotImage(_hiddenActivations[_front], 6, "SFOld:activate:hiddenActivations");
			plots::plotImage(_hiddenStates[_front], 6, "SFOld:activate:hiddenStates");
		}

		//End a simulation step
		void stepEnd() override {

			std::swap(_hiddenStates[_front], _hiddenStates[_back]);
			std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);

			// Swap buffers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
			}
		}

		/*!
		\brief Learning
		\param biasAlpha learning rate of bias.
		*/
		void learn(std::mt19937 &rng) override 
		{
			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				spLearnWeights(
					_hiddenStates[_front],		// in
					vl._derivedInput[_front],	// in
					vl._weights[_back],			// in
					vl._weights[_front],		// out
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					//_activeRatio,				// unused
					vld._weightAlpha);

				std::swap(vl._weights[_front], vl._weights[_back]);
			}

			// Bias update
			spLearnBiases(
				_hiddenSummationTemp[_back],	// in
				//_hiddenStates[_front],		// unused
				_hiddenBiases[_back],			// in
				_hiddenBiases[_front],			// out
				//_activeRatio,					// unused
				_biasAlpha,
				_hiddenSize);

			std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
		}

		/*!
		\brief Inhibit
		Inhibits given activations using this encoder's inhibitory pattern
		*/
		void inhibit(
			const Array2D<float> &activations,
			Array2D<float> &states,
			std::mt19937 &rng) override 
		{
			std::cout << "INFO: SparseFeaturesOld:inhibit" << std::endl;
			// Inhibit
			spInhibit(
				activations,			// in
				states,					// out
				_hiddenSize,
				_inhibitionRadius,
				_activeRatio,
				_hiddenSize
			);
		}
		
		//Get number of visible layers
		size_t getNumVisibleLayers() const {
			return _visibleLayers.size();
		}

		//Get access to visible layer
		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		//Get access to visible layer
		const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		//Get hidden size
		int2 getHiddenSize() const override {
			return _hiddenSize;
		}

		//Get hidden states
		const DoubleBuffer2D<float> &getHiddenStates() const override {
			return _hiddenStates;
		}

		/*!
		\brief Get context
		*/
		const Array2D<float> &getHiddenContext() const override {
			// last checked: 25-nov
			return _hiddenActivations[_back];
		}

		//Clear the working memory
		void clearMemory() override
		{
			clear(_hiddenActivations[_back]);
			clear(_hiddenStates[_back]);

			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				clear(vl._derivedInput[_back]);
			}
		}
		
		// approx memory usage in bytes;
		size_t getMemoryUsage(bool plot) const {
			size_t nBytes = 0;
			size_t bytes;
/*
			bytes = _hiddenActivations[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenActivations:   " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenStates[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenStates:        " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenBiases[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenBiases:        " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenSummationTemp[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenSummationTemp: " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			for (size_t layer = 0; layer < _visibleLayers.size(); ++layer) 
			{
				bytes = _visibleLayers[layer]._derivedInput[0]._data_float.size() * sizeof(float) * 2;
				if (plot) std::cout << "SparseFeatures:_visibleLayers[" << layer << "]:_derivedInput: " << bytes << " bytes" << std::endl;
				nBytes += bytes;

				bytes = _visibleLayers[layer]._weights[0]._data_float.size() * sizeof(float) * 2;
				if (plot) std::cout << "SparseFeatures:_visibleLayers[" << layer << "]:_weights:      " << bytes << " bytes" << std::endl;
				nBytes += bytes;
			}
		*/
			return nBytes;
		}

		static void speedTest(size_t nExperiments = 1) {
#			ifdef _DEBUG
			nExperiments = 1;
#			endif
			speedTest_spLearnWeights(nExperiments);
			speedTest_spStimulus(nExperiments);
		}
		static void speedTest_spLearnWeights(const size_t nExperiments = 1) {
			printf("Running SparseFeatures::speedTest_spLearnWeights\n");
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

			/*
			const int RADIUS = 6;
			const float weightAlpha = 0.002;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Array2D<float> derivedInput = Array2D<float>(visibleSize);
			Array2D<float> hiddenStates = Array2D<float>(hiddenSize);
			Array3D<float> weightsBack = Array3D<float>(weightsSize);
			Array3D<float> weightsFront0 = Array3D<float>(weightsSize);
			Array3D<float> weightsFront1 = Array3D<float>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(derivedInput, visibleSize, initRange, generator);
			randomUniform2D(hiddenStates, hiddenSize, initRange, generator);
			randomUniform3D(weightsBack, weightsSize, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spLearnWeights_v0<RADIUS>(
					hiddenStates,		// in
					derivedInput,		// in
					weightsBack,		// in
					weightsFront0,		// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,		// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[spLearnWeights_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spLearnWeights_v1<RADIUS>(
					hiddenStates,		// in
					derivedInput,		// in
					weightsBack,		// in
					weightsFront1,		// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,		// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[spLearnWeights_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < weightsFront1._size.x; ++x) {
				for (int y = 0; y < weightsFront1._size.y; ++y) {
					for (int z = 0; z < weightsFront1._size.z; ++z) {
						const float f0 = read_3D(weightsFront0, x, y, z);
						const float f1 = read_3D(weightsFront1, x, y, z);
						const float diff = f0 - f1;
						if (std::abs(diff) > 0.000001) printf("WARNING: SparseFeatures::speedTest_spLearnWeights: coord=(%i,%i): f0=%30.28f; f1=%30.28f; diff=%30.28f\n", x, y, f0, f1, diff);
					}
				}
			}
			*/
		}
		static void speedTest_spStimulus(const size_t nExperiments = 1) {
			printf("Running SparseFeatures::speedTest_spStimulus\n");
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));
			/*
			const int RADIUS = 6;
			const int ignoreMiddle = false;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Array2D<float> visibleStates = Array2D<float>(visibleSize);
			Array2D<float> hiddenSummationTempBack = Array2D<float>(hiddenSize);
			Array2D<float> hiddenSummationTempFront0 = Array2D<float>(hiddenSize);
			Array2D<float> hiddenSummationTempFront1 = Array2D<float>(hiddenSize);
			Array3D<float> weights = Array3D<float>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(visibleStates, visibleSize, initRange, generator);
			randomUniform2D(hiddenSummationTempBack, hiddenSize, initRange, generator);
			randomUniform3D(weights, weightsSize, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spStimulus_v0<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront0,	// out
					weights,					// in
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[spStimulus_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spStimulus_v1<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront1,	// out
					weights,					// in
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[spStimulus_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < hiddenSummationTempFront1._size.x; ++x) {
				for (int y = 0; y < hiddenSummationTempFront1._size.y; ++y) {
					const double f0 = read_2D(hiddenSummationTempFront0, x, y);
					const double f1 = read_2D(hiddenSummationTempFront1, x, y);
					const double diff = f0 - f1;
					if (std::abs(diff) > 0.00001) {
						// the floating point model induces some rounding differences between the methods
						printf("WARNING: SparseFeatures::speedTest_spStimulus: coord=(%i,%i): f0=%30.28f; f1=%30.28f; diff=%30.28f\n", x, y, f0, f1, diff);
					}
				}
			}
			*/
		}

		private:

		template <bool CORNER, int RADIUS>
		static void spStimulus_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
#			ifdef USE_FIXED_POINT
				spStimulus_fixedp_kernel<CORNER, RADIUS>(
					hiddenPosition_x,
					hiddenPosition_y,
					visibleStates,
					hiddenSummationTempBack,
					hiddenSummationTempFront,
					weights,
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);
#			else
				spStimulus_floatp_kernel<CORNER, RADIUS>(
					hiddenPosition_x,
					hiddenPosition_y,
					visibleStates,
					hiddenSummationTempBack,
					hiddenSummationTempFront,
					weights,
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);
#			endif
		}

		template <bool CORNER, int RADIUS>
		static void spStimulus_floatp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			float subSum = 0.0f;
			float stateSum = 0.0f;

#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;

					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
					const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
					const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);

					subSum += visibleState * weight;
					stateSum += visibleState;
				}
			}

			if (ignoreMiddle) { // substract the visible state that corresponds to dx=dy=0

				const int visiblePosition_x = visiblePositionCenter_x;
				const int visiblePosition_y = visiblePositionCenter_y;

				const int offset_x = visiblePositionCenter_x - fieldLowerBound_x;
				const int offset_y = visiblePositionCenter_y - fieldLowerBound_y;

				const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
				const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
				const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);

				subSum -= visibleState * weight;
				stateSum -= visibleState;
			}

			const float stimulusAddition = subSum / std::max(0.0001f, stateSum);
			const float oldValue = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
			const float newValue = oldValue + stimulusAddition;
			//std::cout << "SparseFeatures::spStimulus_float_kernel: stimulusAddition=" << stimulusAddition << "; oldValue=" << oldValue << "; newValue=" << newValue << std::endl;
			write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newValue);
		}

		template <bool CORNER, int RADIUS>
		static void spStimulus_fixedp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			FixedP3 subSum = 0;
			FixedP3 stateSum = 0; // FixPoint2 should also provide sufficient room

#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;
					
					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
					const FixedP weight = read_3D_fixp(weights, hiddenPosition_x, hiddenPosition_y, wi);
					const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);

					subSum += static_cast<FixedP3>(visibleState) * static_cast<FixedP3>(weight);
					stateSum += static_cast<FixedP3>(visibleState);
				}
			}

			if (ignoreMiddle) { // substract the visible state that corresponds to dx=dy=0

				const int visiblePosition_x = visiblePositionCenter_x;
				const int visiblePosition_y = visiblePositionCenter_y;

				const int offset_x = visiblePosition_x - fieldLowerBound_x;
				const int offset_y = visiblePosition_y - fieldLowerBound_y;

				const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
				const FixedP weight = read_3D_fixp(weights, hiddenPosition_x, hiddenPosition_y, wi);
				const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);

				subSum -= static_cast<FixedP3>(visibleState) * static_cast<FixedP3>(weight);
				stateSum -= static_cast<FixedP3>(visibleState);
			}

			const FixedP sumFixP = read_2D_fixp(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
			FixedP newState;
			if (stateSum == 0) {
				newState = sumFixP;
			} else {
				//const float subSumF = toFloat(static_cast<FixedP2>(subSum));
				//const float stateSumF = toFloat(static_cast<FixedP>(stateSum));

				const float subSumF = static_cast<float>(subSum) / (1 << (N_BITS_DENOMINATOR * 2));
				const float stateSumF = static_cast<float>(stateSum) / (1 << N_BITS_DENOMINATOR);

				const float stimulusAdditionF = subSumF / stateSumF;

				newState = add_saturate(sumFixP, toFixedP(stimulusAdditionF));
			}
			write_2D_fixp(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newState);
		}

		template <int RADIUS>
		static void spStimulus_v0(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			for (int x = 0; x < hiddenSummationTempBack._size.x; ++x) {
				for (int y = 0; y < hiddenSummationTempBack._size.y; ++y) {
					spStimulus_kernel<true, RADIUS>(x, y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
		}

		template <int RADIUS>
		static void spStimulus_v1(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			std::tuple<int2, int2> ranges = cornerCaseRange(hiddenSummationTempBack._size, visibleStates._size, RADIUS, hiddenToVisible);
			const int x0 = 0;
			const int x1 = std::get<0>(ranges).x;
			const int x2 = std::get<0>(ranges).y;
			const int x3 = hiddenSummationTempBack._size.x;
			const int y0 = 0;
			const int y1 = std::get<1>(ranges).x;
			const int y2 = std::get<1>(ranges).y;
			const int y3 = hiddenSummationTempBack._size.y;

			for (int hiddenPosition_x = x0; hiddenPosition_x < x1; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					spStimulus_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
		}

		static void spStimulus(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const bool ignoreMiddle) 
		{
			switch (radius) {
			case 6: spStimulus_v0<6>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle); break;
			case 8: spStimulus_v0<8>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle); break;
			case 20: spStimulus_v0<20>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle); break;
			default: printf("ERROR: SparseFeatures::spStimulus: provided radius %i is not implemented\n", radius); break;
			}
		}

		static void spActivate(
			const Array2D<float> &stimuli,
			//const Array2D<float> /*&hiddenStates*/,
			const Array2D<float> &biases,
			//const Array2D<float> /*&hiddenActivationsBack*/,
			Array2D<float> &hiddenActivationsFront, // write only
			const int2 range)
		{
			if (true) { //TODO
				const int nElements = range.x * range.y;
				for (int i = 0; i < nElements; ++i) {
					const float stimulus = stimuli._data_float[i];
					const float bias = biases._data_float[i];
					const float activation = stimulus + bias;
					hiddenActivationsFront._data_float[i] = activation;
#					ifdef USE_FIXED_POINT
					hiddenActivationsFront._data_fixP[i] = toFixedP(activation);
#					endif
				}
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {

						const float stimulus = read_2D(stimuli, x, y);
						//const float activationPrev = read_imagef_2D(hiddenActivationsBack, x, y);
						//const float statePrev = read_imagef_2D(hiddenStates, x, y);
						const float bias = read_2D(biases, x, y);
						const float activation = stimulus + bias;
						write_2D(hiddenActivationsFront, x, y, activation);
					}
				}
			}
		}

		static void spInhibit(
			const Array2D<float> &activations,
			Array2D<float> &hiddenStatesFront, // write only
			const int2 hiddenSize, 
			const int radius, 
			const float activeRatio,
			const int2 range)
		{
			for (int x = 0; x < range.x; ++x) {
				for (int y = 0; y < range.y; ++y) {

					const float activation = read_2D(activations, x, y);
					int inhibition = 0;
					int count = 0;

					//TODO optimize boundary condition


#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int otherPosition_x = x + dx;

						if (inBounds(otherPosition_x, hiddenSize.x)) {
#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) {
								if (dx == 0 && dy == 0) {
									// do nothing
								}
								else {
									const int otherPosition_y = y + dy;
									if (inBounds(otherPosition_y, hiddenSize.y)) {
										const float otherActivation = read_2D(activations, otherPosition_x, otherPosition_y);
										inhibition += (otherActivation >= activation) ? 1 : 0;
										count++;
									}
								}
							}
						}
					}
					const float state = (inhibition < (activeRatio * count)) ? 1.0f : 0.0f;
					write_2D(hiddenStatesFront, x, y, state);
				}
			}
		}

		template <bool CORNER, int RADIUS>
		static void spLearnWeights_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const float weightAlpha,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront)
		{
#			ifdef USE_FIXED_POINT
				spLearnWeights_fixedp_kernel<CORNER, RADIUS>(
					hiddenPosition_x,
					hiddenPosition_y,
					visibleSize,
					hiddenToVisible,
					weightAlpha,
					hiddenStates,
					visibleStates,
					weightsBack,
					weightsFront);
#			else
				spLearnWeights_floatp_kernel<CORNER, RADIUS>(
					hiddenPosition_x,
					hiddenPosition_y,
					visibleSize,
					hiddenToVisible,
					weightAlpha,
					hiddenStates,
					visibleStates,
					weightsBack,
					weightsFront);
#			endif
		}

		template <bool CORNER, int RADIUS>
		static void spLearnWeights_floatp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const float weightAlpha,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront)
		{
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			const float hiddenState = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
			const float hiddenStateWeightAlpha = weightAlpha * hiddenState;
			//std::printf("INFO: spLearnWeights_fixedp_kernel: weightAlpha=%24.22f; hiddenState=%24.22f\n", weightAlpha, hiddenState);

#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;

					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
					const float weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
					const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
					const float wD = hiddenStateWeightAlpha * (visibleState - weightPrev);
					const float weight = weightPrev + wD;
					const float weightSaturated = (weight > 1.0f) ? 1.0f : ((weight < 0.0f) ? 0.0f : weight);
					write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weightSaturated);
				}
			}
		}

		template <bool CORNER, int RADIUS>
		static void spLearnWeights_fixedp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const float weightAlpha,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront)
		{
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			const FixedP weightAlphaFixP = toFixedP(weightAlpha);
			if (weightAlphaFixP == 0) std::printf("WARNING: spLearnWeights_fixedp_kernel: weightAlphaFixP is zero\n");

			const FixedP hiddenState = read_2D_fixp(hiddenStates, hiddenPosition_x, hiddenPosition_y);
			const FixedP hiddenStateWeightAlpha = multiply_saturate(hiddenState, weightAlphaFixP);

			std::printf("INFO: spLearnWeights_fixedp_kernel: hiddenStateWeightAlpha=%i\n", static_cast<int>(hiddenStateWeightAlpha));


#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;
					
					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));

					const FixedP weightPrev = read_3D_fixp(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
					const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);
					FixedP weight;
					if (visibleState > weightPrev)
					{
						const FixedP weightDelta = multiply_saturate(hiddenStateWeightAlpha, (visibleState - weightPrev));
						weight = add_saturate(weightPrev, weightDelta);
						//std::cout << "INFO: spLearnWeights_fixedp_kernel: A: weight_fixP=" << toFloat(weight) << "; weight_old=" << weight_old << std::endl;
					} else
					{
						const FixedP weightDelta = multiply_saturate(hiddenStateWeightAlpha, (weightPrev - visibleState));
						weight = substract_saturate(weightPrev, weightDelta);
						//std::cout << "INFO: spLearnWeights_fixedp_kernel: B: weight_fixP=" << toFloat(weight) << "; weight_old="<< weight_old << std::endl;
					}
					write_3D_fixp(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weight);
				}
			}
		}

		template <int RADIUS>
		static void spLearnWeights_v1(
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, // write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			//const float /*activeRatio*/, //unused
			const float weightAlpha)
		{
			std::tuple<int2, int2> ranges = cornerCaseRange(hiddenStates._size, visibleStates._size, RADIUS, hiddenToVisible);
			const int x0 = 0;
			const int x1 = std::get<0>(ranges).x;
			const int x2 = std::get<0>(ranges).y;
			const int x3 = hiddenStates._size.x;
			const int y0 = 0;
			const int y1 = std::get<1>(ranges).x;
			const int y2 = std::get<1>(ranges).y;
			const int y3 = hiddenStates._size.y;

			for (int hiddenPosition_x = x0; hiddenPosition_x < x1; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					spLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					spLearnWeights_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
			}
		}

		template <int RADIUS>
		static void spLearnWeights_v0(
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, // write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			//const float /*activeRatio*/, //unused
			const float weightAlpha)
		{
			for (int hiddenPosition_x = 0; hiddenPosition_x < hiddenStates._size.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < hiddenStates._size.y; ++hiddenPosition_y) {
					spLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleSize, hiddenToVisible, weightAlpha, hiddenStates, visibleStates, weightsBack, weightsFront);
				}
			}
		}

		static void spLearnWeights(
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, // write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			//const float /*activeRatio*/, //unused
			const float weightAlpha)
		{
			//printf("hiddenStates.size=(%i,%i)\n", hiddenStates._size.x, hiddenStates._size.y);
			//printf("visibleStates.size=(%i,%i)\n", visibleStates._size.x, visibleStates._size.y);
			//printf("weightsBack.size=(%i,%i,%i)\n", weightsBack._size.x, weightsBack._size.y, weightsBack._size.z);
			//printf("hiddenToVisible=(%f,%f)\n", hiddenToVisible.x, hiddenToVisible.y);

			switch (radius) {
				case 6: spLearnWeights_v1<6>(hiddenStates, visibleStates, weightsBack, weightsFront, visibleSize, hiddenToVisible, weightAlpha); break;
				case 8: spLearnWeights_v1<8>(hiddenStates, visibleStates, weightsBack, weightsFront, visibleSize, hiddenToVisible, weightAlpha); break;
				case 20: spLearnWeights_v1<20>(hiddenStates, visibleStates, weightsBack, weightsFront, visibleSize, hiddenToVisible, weightAlpha); break;
				default: printf("ERROR: SparseFeatures::spLearnWeights: provided radius %i is not implemented\n", radius); break;
			}
		}

		static void spLearnBiases(
			const Array2D<float> &stimuli,
			//const Array2D<float> /*&hiddenStates*/, // unused
			const Array2D<float> &hiddenBiasesBack,
			Array2D<float> &hiddenBiasesFront, //write only
			//const float /*activeRatio*/, // unused
			const float biasAlpha,
			const int2 range)
		{
			if (true) { //TODO
				const int nElements = range.x * range.y;
				if (UPDATE_FLOATING_POINT) {
#					pragma ivdep
					for (int i = 0; i < nElements; ++i) {
						const float stimulus = stimuli._data_float[i];
						const float hiddenBiasPrev = hiddenBiasesBack._data_float[i];
						//INFO: HiddenBiases can be negative
						hiddenBiasesFront._data_float[i] = hiddenBiasPrev + (biasAlpha * (-stimulus - hiddenBiasPrev));
					}
				}
#				ifdef USE_FIXED_POINT
				const FixedP biasAlphaFP = toFixedP(biasAlpha);
#				pragma ivdep
				for (int i = 0; i < nElements; ++i) {
					const FixedP stimulus = stimuli._data_fixP[i];
					const FixedP hiddenBiasPrev = hiddenBiasesBack._data_fixP[i];
					const int hiddenBiasInt = static_cast<int>(biasAlphaFP) * (-static_cast<int>(stimulus) - hiddenBiasPrev);
					const FixedP hiddenBias = (hiddenBiasInt < 0)
						? substract_saturate(hiddenBiasPrev, toFixedP(-hiddenBiasInt))
						: add_saturate(hiddenBiasPrev, toFixedP(hiddenBiasInt));
					hiddenBiasesFront._data_fixP[i] = hiddenBias;
				}
#				endif
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {
						const float stimulus = read_2D(stimuli, x, y);
						//float hiddenState = read_imagef_2D(hiddenStates, hiddenPosition); //TODO: unused
						const float hiddenBiasPrev = read_2D(hiddenBiasesBack, x, y);
						write_2D(hiddenBiasesFront, x, y, hiddenBiasPrev + (biasAlpha * (-stimulus - hiddenBiasPrev)));
					}
				}
			}
		}

		static void spDeriveInputs(
			const Array2D<float> &inputs,
			//const Array2D<float> /*&outputsBack*/, // unused
			Array2D<float> &outputsFront, // write only
			const int2 range)
		{
			if (true) {
				copy(inputs, outputsFront);
			}
			else {
#				pragma ivdep 
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep 
					for (int y = 0; y < range.y; ++y) {
						const float input = read_2D(inputs, x, y);
						write_2D(outputsFront, x, y, input);
					}
				}
			}
		}
	};
}

