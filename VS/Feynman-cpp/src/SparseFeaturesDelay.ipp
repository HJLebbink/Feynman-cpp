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
			DoubleBuffer2D<float> _derivedInput;

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

			std::string info() const override {
				std::string result = "\n";
				for (size_t i = 0; i < _visibleLayerDescs.size(); ++i) {
					result += "visibleLayer[" + std::to_string(i) + "]: size=(" + std::to_string(_visibleLayerDescs[i]._size.x) + "," + std::to_string(_visibleLayerDescs[i]._size.y) + ")\n";
					result += "visibleLayer[" + std::to_string(i) + "]: radius=" + std::to_string(_visibleLayerDescs[i]._radius) + "\n";
					result += "visibleLayer[" + std::to_string(i) + "]: weightAlpha=" + std::to_string(_visibleLayerDescs[i]._weightAlpha) + "\n";
					result += "visibleLayer[" + std::to_string(i) + "]: ignoreMiddle=" + std::to_string(_visibleLayerDescs[i]._ignoreMiddle) + "\n";
				}
				result += "TODO\n";
				/*				result += "hiddenSize=(" + std::to_string(_hiddenSize.x) + "," + std::to_string(_hiddenSize.y) + ")\n";;
				result += "inhibitionRadius=" + std::to_string(_inhibitionRadius) + "\n";
				result += "activeRatio=" + std::to_string(_activeRatio) + "\n";
				result += "biasAlpha=" + std::to_string(_biasAlpha) + "\n";
				result += "initWeightRange=(" + std::to_string(_initWeightRange.x) + "," + std::to_string(_initWeightRange.y) + ")\n";
				result += "initBiasRange=(" + std::to_string(_initBiasRange.x) + "," + std::to_string(_initBiasRange.y) + ")\n";
				*/				return result;
			}

			//Factory
			std::shared_ptr<SparseFeatures> sparseFeaturesFactory() override {
				if (true) std::cout << "INFO: SFDelay:sparseFeaturesFactory:" << info() << std::endl;
				return std::make_shared<SparseFeaturesDelay>(_visibleLayerDescs, _hiddenSize, _inhibitionRadius, _biasAlpha, _activeRatio, _gamma, _initWeightRange, _rng);
			}
		};

	private:

		//Activations, states, biases
		DoubleBuffer2D<float> _hiddenActivations;
		DoubleBuffer2D<float> _hiddenStates;
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
			_hiddenStates = createDoubleBuffer2D<float>(_hiddenSize);
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
			const std::vector<Array2D<float>> &visibleStates,
			const Array2D<float> &predictionsPrev,
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
			sfdInhibit(
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
		const DoubleBuffer2D<float> &getHiddenStates() const override {
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

	private:

		static void sfdStimulus(
			const Array2D<float> &visibleStates,			// in
			const Array2D<float> &hiddenSummationTempBack,	// in
			Array2D<float> &hiddenSummationTempFront,		// write only
			const Array3D<WEIGHT_T> &weights,				// in
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
									const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi).x;
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
			const Array2D<float> &stimuli,					// in
			const Array2D<float> &hiddenStates,				// in
			const Array2D<float> &biases,					// in
			const Array2D<float> &hiddenActivationsBack,	// in
			Array2D<float> &hiddenActivationsFront,			// out write only
			const int2 range)
		{
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float stimulus = stimuli._data_float[i];
				const float activationPrev = hiddenActivationsBack._data_float[i];
				const float statePrev = hiddenStates._data_float[i];
				const float bias = biases._data_float[i];
				const float activation = fmax(0.0f, activationPrev * (1.0f - statePrev) + stimulus + bias);
				hiddenActivationsFront._data_float[i] = activation;
			}
		}

		static void sfdInhibit(
			const Array2D<float> &activations,	// in
			Array2D<float> &hiddenStatesFront,	// out: write only
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

		static void sfdLearnWeights(
			const Array2D<float> &hiddenStates,			// in
			const Array2D<float> &hiddenStatesPrev,		// in
			const Array2D<float> &visibleStates,		// in
			const Array2D<float> &visibleStatesPrev,	// in
			const Array3D<WEIGHT_T> &weightsBack,		// in
			Array3D<WEIGHT_T> &weightsFront,			// out: write only
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

					const float hiddenState = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
					const float hiddenStatePrev = read_2D(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y);

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
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
									const float visibleStatePrev = read_2D(visibleStatesPrev, visiblePosition_x, visiblePosition_y);

									const float visibleState_y = 0; //TODO

									const float traceShort = weightPrev.y * lambda + (1.0f - lambda) * hiddenState * visibleState_y;
									const float traceLong = weightPrev.z * gamma + (1.0f - gamma) * hiddenState * visibleState_y;
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
			const Array2D<float> &stimuli,			// in
			const Array2D<float> &hiddenStates,		// in
			const Array2D<float> &hiddenBiasesBack,	// in
			Array2D<float> &hiddenBiasesFront,		// out: write only
			const float activeRatio,
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
			const Array2D<float> &inputs,	// in
			Array2D<float> &outputsFront,	// out: write only
			const int2 range)
		{
			copy(inputs, outputsFront);
		}
	};
}