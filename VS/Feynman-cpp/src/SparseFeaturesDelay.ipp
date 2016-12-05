// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once
#include <memory>
#include "Helpers.ipp"
#include "SparseFeatures.ipp"
#include "FixedPoint.ipp"


namespace feynman {

	/*!
	\brief STDP encoder (sparse features)
	Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
	*/
	class SparseFeaturesDelay : public SparseFeatures {
	public:

		using WEIGHT_T = float3;


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

			//Input decay
			float _lambda;

			//trace decay
			float _gamma;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(6), _ignoreMiddle(false),
				_weightAlpha(0.0001f), _lambda(0.9f), _gamma(0.96f)
			{}
		};

		//Visible layer
		struct VisibleLayer {
			//Possibly manipulated input
			DoubleBuffer2D<float2> _derivedInput;

			//Weights
			DoubleBuffer3D<WEIGHT_T> _weights; // Encoding weights (creates spatio-temporal sparse code)

			float2 _hiddenToVisible;
			float2 _visibleToHidden;

			int2 _reverseRadii;
		};

		//Sparse Features Chunk Descriptor
		class SparseFeaturesDelayDesc : public SparseFeatures::SparseFeaturesDesc {
		public:
			std::vector<VisibleLayerDesc> _visibleLayerDescs;
			int2 _hiddenSize;
			int _inhibitionRadius;
			float _biasAlpha;
			float _activeRatio;
			float _gamma;
			float2 _initWeightRange;
			std::mt19937 _rng;

			//Defaults
			SparseFeaturesDelayDesc()
				: _hiddenSize({ 16, 16 }),
				_inhibitionRadius(6),
				_biasAlpha(0.01f), _activeRatio(0.04f), _gamma(0.9f),
				_initWeightRange({-0.1f, 0.01f }),
				_rng()
			{
				_name = "delay";
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
				return std::make_shared<SparseFeaturesDelay>(_visibleLayerDescs, _hiddenSize, _inhibitionRadius, _biasAlpha, _activeRatio, _gamma, _initWeightRange, _rng);
			}
		};

	private:

		//Activations, states, biases
		DoubleBuffer2D<float> _hiddenActivations;
		DoubleBuffer2D<float2> _hiddenStates;
		DoubleBuffer2D<float> _hiddenBiases;

		int2 _hiddenSize;
		int _inhibitionRadius;

		//Hidden summation temporary buffer
		DoubleBuffer2D<float> _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

	public:
		float _activeRatio;
		float _biasAlpha;

		//Default constructor
		SparseFeaturesDelay() {};

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information.
		\param visibleLayerDescs descriptors for each input layer.
		\param hiddenSize hidden layer (SDR) size (2D).
		\param rng a random number generator.
		*/
		SparseFeaturesDelay(
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const int2 hiddenSize,
			const int inhibitionRadius,
			const float biasAlpha,
			const float activeRatio,
			const float gamma,
			const float2 initWeightRange,
			std::mt19937 &rng
		) :
			_hiddenSize(hiddenSize),
			_inhibitionRadius(inhibitionRadius),
			_activeRatio(activeRatio),
			_biasAlpha(biasAlpha)
		{
			_type = SparseFeaturesType::_delay;
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
					vl._weights = createDoubleBuffer3D<WEIGHT_T>(weightsSize);
					randomUniform3D(vl._weights[_back], initWeightRange, rng);
				}
				vl._derivedInput = createDoubleBuffer2D<float>(vld._size);
				clear(vl._derivedInput[_back]);
			}

			// Hidden state data
			_hiddenActivations = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenStates = createDoubleBuffer2D<float2>(_hiddenSize);
			_hiddenBiases = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenSummationTemp = createDoubleBuffer2D<float>(_hiddenSize);

			clear(_hiddenActivations[_back]);
			clear(_hiddenStates[_back]);

			randomUniform2D(_hiddenBiases[_back], initWeightRange, rng);
		}


		/*!
		\brief Activate predictor
		\param lambda decay of hidden unit traces.
		\param activeRatio % active units.
		\param rng a random number generator.
		*/
		void activate(
			const std::vector<Array2D<float2>> &visibleStates,
			const Array2D<float2> &predictionsPrev,
			std::mt19937 &rng) override
		{
			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				sfdDeriveInputs(
					visibleStates[vli],			// in
					vl._derivedInput[_front],	// out
					vld._size);

				sfdStimulus(
					vl._derivedInput[_front],		// in
					_hiddenSummationTemp[_back],	// in
					_hiddenSummationTemp[_front],	// out
					vl._weights[_back],				// in
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					vld._ignoreMiddle,
					_hiddenSize);

				// Swap buffers
				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}

			// Activate
			sfdActivate(
				_hiddenSummationTemp[_back],	// in
				_hiddenStates[_back],			// unused
				_hiddenBiases[_back],			// in
				_hiddenActivations[_back],		// unused
				_hiddenActivations[_front],		// out
				_hiddenSize);

			// Inhibit
			sfdInhibit(
				_hiddenActivations[_front],		// in
				_hiddenStates[_back],			// ?
				_hiddenSize,
				_inhibitionRadius,
				_activeRatio,
				_hiddenSize);
		}
		
		//End a simulation step
		void stepEnd() override
		{
			std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
			std::swap(_hiddenStates[_front], _hiddenStates[_back]);

			// Swap buffers
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
			}
		}

		/*!
		\brief Learning
		\param biasAlpha learning rate of bias.
		\param activeRatio % active units.
		\param gamma synaptic trace decay.
		*/
		void learn(std::mt19937 &rng) override {
			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				// Weight update
				sfdLearnWeights(
					_hiddenStates[_front],
					_hiddenStates[_back],
					vl._derivedInput[_front],
					vl._derivedInput[_back],
					vl._weights[_back],
					vl._weights[_front],
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					_activeRatio,
					vld._weightAlpha,
					vld._lambda,
					vld._gamma,
					_hiddenSize
				);
				std::swap(vl._weights[_front], vl._weights[_back]);
			}

			// Bias update
			sfdLearnBiases(
				_hiddenSummationTemp[_back],
				_hiddenStates[_front],
				_hiddenBiases[_back],
				_hiddenBiases[_front],
				_activeRatio,
				_biasAlpha,
				_hiddenSize
			);
			std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);

		}

		/*!
		\brief Inhibit
		Inhibits given activations using this encoder's inhibitory pattern
		*/
		void inhibit(const Array2D<float> &activations, Array2D<float> &states, std::mt19937 &rng) override {
			sfdInhibitPred(
				activations,
				states,
				_hiddenSize,
				_inhibitionRadius,
				_activeRatio,
				_hiddenSize
			);
		}

		/*!
		\brief Get number of visible layers
		*/
		size_t getNumVisibleLayers() const {
			return _visibleLayers.size();
		}

		/*!
		\brief Get access to visible layer
		*/
		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		/*!
		\brief Get access to visible layer
		*/
		const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		/*!
		\brief Get hidden size
		*/
		int2 getHiddenSize() const override {
			return _hiddenSize;
		}

		/*!
		\brief Get hidden states
		*/
		const DoubleBuffer2D<float2> &getHiddenStates() const override {
			return _hiddenStates;
		}

		/*!
		\brief Clear the working memory
		*/
		void clearMemory() override {
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
			return nBytes;
			*/
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

			const int RADIUS = 6;
			const float weightAlpha = 0.002;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Array2D<float> derivedInput = Array2D<float>(visibleSize);
			Array2D<float> hiddenStates = Array2D<float>(hiddenSize);
			Array3D<WEIGHT_T> weightsBack = Array3D<WEIGHT_T>(weightsSize);
			Array3D<WEIGHT_T> weightsFront0 = Array3D<WEIGHT_T>(weightsSize);
			Array3D<WEIGHT_T> weightsFront1 = Array3D<WEIGHT_T>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(derivedInput, initRange, generator);
			randomUniform2D(hiddenStates, initRange, generator);
			randomUniform3D(weightsBack, initRange, generator);

			/*
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
			Array3D<WEIGHT_T> weights = Array3D<WEIGHT_T>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(visibleStates, initRange, generator);
			randomUniform2D(hiddenSummationTempBack, initRange, generator);
			randomUniform3D(weights, initRange, generator);

			/*
			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				sfsStimulus_v0<RADIUS>(
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

		static void sfdStimulus(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<WEIGHT_T> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const bool ignoreMiddle,
			const int2 range)
		{
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {
					
					const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
					const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);

					float subSum = 0.0f;
					float stateSum = 0.0f;

					const int fieldLowerBound_x = visiblePositionCenter_x - radius;
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							if (ignoreMiddle && dx == 0 && dy == 0)
								continue;

							const int visiblePosition_x = visiblePositionCenter_x + dx;
							const int visiblePosition_y = visiblePositionCenter_y + dy;

							if (inBounds(visiblePosition_x, visibleSize.x)) {
								if (inBounds(visiblePosition_x, visibleSize.x)) {
									const int offset_x = visiblePosition_x - fieldLowerBound_x;
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + offset_x * (radius * 2 + 1);
									const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);

									subSum += weight * visibleState;
									stateSum += visibleState * visibleState;
								}
							}
						}
					}
					const float sum = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
					const float newValue = sum + subSum / fmax(0.0001f, sqrt(stateSum));
					write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newValue);
				}
			}
		}

		static void sfdActivate(
			const Array2D<float> &stimuli,
			const Array2D<float2> &hiddenStates,
			const Array2D<float> &biases,
			const Array2D<float> &hiddenActivationsBack,
			Array2D<float> &hiddenActivationsFront,		// write only
			const int2 range)
		{
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float stimulus = stimuli._data_float[i];
				const float activationPrev = hiddenActivationsBack._data_float[i];
				const float statePrev = hiddenStates._data_float[i].x;
				const float bias = biases._data_float[i];
				const float activation = fmax(0.0f, activationPrev * (1.0f - statePrev) + stimulus + bias);
				hiddenActivationsFront._data_float[i] = activation;
			}
		}

		static void sfdInhibit(
			const Array2D<float> &activations,
			Array2D<float> &hiddenStatesFront, // write only
			const int2 hiddenSize,
			const int radius,
			const float activeRatio,
			const int2 range)
		{
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {

					const float activation = read_2D(activations, hiddenPosition_x, hiddenPosition_y);
					int inhibition = 0;
					int count = 0;

#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int otherPosition_x = hiddenPosition_x + dx;

						if (inBounds(otherPosition_x, hiddenSize.x)) {
#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) {
								if (dx == 0 && dy == 0) {
									// do nothing
								}
								else {
									const int otherPosition_y = hiddenPosition_y + dy;
									if (inBounds(otherPosition_y, hiddenSize.y)) {
										const float otherActivation = read_2D(activations, otherPosition_x, otherPosition_y);
										if (otherActivation >= activation) inhibition++;
										count++;
									}
								}
							}
						}
					}
					const float state = (inhibition < (activeRatio * count)) ? 1.0f : 0.0f;
					write_2D(hiddenStatesFront, hiddenPosition_x, hiddenPosition_y, state);
				}
			}
		}

		template <bool CORNER, int RADIUS>
		static void sfdLearnWeights_floatp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const float weightAlpha,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &visibleStates,
			const Array3D<WEIGHT_T> &weightsBack,
			Array3D<WEIGHT_T> &weightsFront)
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

		static void sfdLearnWeights(
			const Array2D<float2> &hiddenStates,
			const Array2D<float2> &hiddenStatesPrev,
			const Array2D<float2> &visibleStates,
			const Array2D<float2> &visibleStatesPrev,
			const Array3D<WEIGHT_T> &weightsBack,
			Array3D<WEIGHT_T> &weightsFront, // write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const float activeRatio, //unused
			const float weightAlpha,
			const float lambda,
			const float gamma,
			const int2 range)
		{
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {
					const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
					const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);

					const int fieldLowerBound_x = visiblePositionCenter_x - radius;
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					const float2 hiddenState = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
					const float2 hiddenStatePrev = read_2D(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y);

					float weightSum = 0.0f;

					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							const int visiblePosition_x = visiblePositionCenter_x + dx;
							const int visiblePosition_y = visiblePositionCenter_y + dy;

							if (inBounds(visiblePosition_x, visibleSize.x)) {
								if (inBounds(visiblePosition_x, visibleSize.x)) {
									const int offset_x = visiblePosition_x - fieldLowerBound_x;
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + offset_x * (radius * 2 + 1);
									const WEIGHT_T weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);

									weightSum += weightPrev.x * weightPrev.x;
								}
							}
						}
					}
					const float scale = 1.0f / fmax(0.0001f, sqrt(weightSum));

					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							const int visiblePosition_x = visiblePositionCenter_x + dx;
							const int visiblePosition_y = visiblePositionCenter_y + dy;

							if (inBounds(visiblePosition_x, visibleSize.x)) {
								if (inBounds(visiblePosition_x, visibleSize.x)) {
									const int offset_x = visiblePosition_x - fieldLowerBound_x;
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + offset_x * (radius * 2 + 1);

									const WEIGHT_T weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
									const float2 visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
									const float2 visibleStatePrev = read_2D(visibleStatesPrev, visiblePosition_x, visiblePosition_y);

									const float traceShort = weightPrev.y * lambda + (1.0f - lambda) * hiddenState.x * visibleState.y;
									const float traceLong = weightPrev.z * gamma + (1.0f - gamma) * hiddenState.x * visibleState.y;
									const float learn = traceLong - traceShort;

									const WEIGHT_T newValue = { weightPrev.x * scale + weightAlpha * learn, traceShort, traceLong };
									write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, newValue);
								}
							}
						}
					}
				}
			}
		}

		static void sfdLearnBiases(
			const Array2D<float> &stimuli,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &hiddenBiasesBack,
			Array2D<float> &hiddenBiasesFront, //write only
			const float biasAlpha,
			const int2 range)
		{
			const int nElements = range.x * range.y;
#			pragma ivdep
			for (int i = 0; i < nElements; ++i) {
				const float stimulus = stimuli._data_float[i];
				//const float hiddenState = hiddenStates._data_float[i];
				const float hiddenBiasPrev = hiddenBiasesBack._data_float[i];
				const float newValue = hiddenBiasPrev + (biasAlpha * (-stimulus - hiddenBiasPrev));
				hiddenBiasesFront._data_float[i] = newValue;
			}
		}

		static void sfdDeriveInputs(
			const Array2D<float2> &inputs,
			Array2D<float2> &outputsFront, // write only
			const int2 range)
		{
			copy(inputs, outputsFront);
		}
	};
}