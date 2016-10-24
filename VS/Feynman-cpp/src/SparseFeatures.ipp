// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <tuple>
#include <vector>
#include <random>
#include <algorithm> // for std::min

#include "timing.h"
#include "Helpers.ipp"

namespace feynman {

	//Sparse predictor
	//Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
	class SparseFeatures {
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
			DoubleBuffer2D _derivedInput;

			DoubleBuffer3D _weights;
			float2 _hiddenToVisible;
			float2 _visibleToHidden;
			int2 _reverseRadii;
		};

	private:
		// Hidden activations, states, biases, errors, predictions
		DoubleBuffer2D _hiddenActivations;
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenBiases;

		//Hidden size
		int2 _hiddenSize;

		//Lateral inhibitory radius
		int _inhibitionRadius;

		//Hidden summation temporary buffer
		DoubleBuffer2D _hiddenSummationTemp;

		//Layers and descs
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization.
		Requires initialization information.
		\param visibleLayerDescs descriptors for each input layer.
		\param hiddenSize hidden layer (SDR) size (2D).
		\param inhibitionRadius inhibitory radius.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const int2 hiddenSize,
			const int inhibitionRadius,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			_visibleLayerDescs = visibleLayerDescs;
			_hiddenSize = hiddenSize;
			_inhibitionRadius = inhibitionRadius;

			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (int vli = 0; vli < static_cast<int>(_visibleLayers.size()); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				vl._hiddenToVisible = float2{ 
					static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
					static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
				};
				vl._visibleToHidden = float2{ 
					static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
					static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
				};
				vl._reverseRadii = int2{ 
					static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
				};
				{
					int weightDiam = vld._radius * 2 + 1;
					int numWeights = weightDiam * weightDiam;
					int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._weights = createDoubleBuffer3D(weightsSize);
					randomUniform3D(vl._weights[_back], weightsSize, { 0.0f, 1.0f }, rng);
				}

				vl._derivedInput = createDoubleBuffer2D(vld._size);
				clear(vl._derivedInput[_back]);
			}

			// Hidden state data
			_hiddenActivations = createDoubleBuffer2D(_hiddenSize);
			_hiddenStates = createDoubleBuffer2D(_hiddenSize);
			_hiddenBiases = createDoubleBuffer2D(_hiddenSize);
			_hiddenSummationTemp = createDoubleBuffer2D(_hiddenSize);

			clear(_hiddenActivations[_back]);
			clear(_hiddenStates[_back]);

			randomUniform2D(_hiddenBiases[_back], _hiddenSize, initWeightRange, rng);
		}


		/*!
		\brief Activate predictor
		\param visibleStates the input layer states.
		\param activeRatio % active units.
		\param rng a random number generator.
		*/
		void activate(
			const std::vector<Image2D> &visibleStates,
			const float activeRatio, 
			const std::mt19937 /*&rng*/) 
		{
			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				spDeriveInputs(
					visibleStates[vli],			// in
					//vl._derivedInput[_back],	// unused
					vl._derivedInput[_front],	// out
					vld._size);

				spStimulus(
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
			spActivate(
				_hiddenSummationTemp[_back],	// in
				//_hiddenStates[_back],			// unused
				_hiddenBiases[_back],			// in
				//_hiddenActivations[_back],	// unused
				_hiddenActivations[_front],		// out
				_hiddenSize);

			// Inhibit
			spInhibit(
				_hiddenActivations[_front],		// in
				_hiddenStates[_front],			// out
				_hiddenSize,
				_inhibitionRadius,
				activeRatio,
				_hiddenSize);
		}

		//End a simulation step
		void stepEnd() {

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
		void learn(
			const float biasAlpha, 
			const float /*activeRatio*/)
		{
			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				spLearnWeights(
					_hiddenStates[_front],		// in
					vl._derivedInput[_front],	// in
					vl._weights[_back],			// in
					vl._weights[_front],		// out
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					//activeRatio,				// unused
					vld._weightAlpha,
					_hiddenSize);

				std::swap(vl._weights[_front], vl._weights[_back]);
			}

			// Bias update
			spLearnBiases(
				_hiddenSummationTemp[_back],	// in
				//_hiddenStates[_front],		// unused
				_hiddenBiases[_back],			// in
				_hiddenBiases[_front],			// out
				//activeRatio,					// unused
				biasAlpha,
				_hiddenSize);

			std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
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
		int2 getHiddenSize() const {
			return _hiddenSize;
		}

		//Get hidden states
		const DoubleBuffer2D &getHiddenStates() const {
			return _hiddenStates;
		}

		//Get hidden biases
		const DoubleBuffer2D &getHiddenBiases() const {
			return _hiddenBiases;
		}

		//Clear the working memory
		void clearMemory() 
		{
			clear(_hiddenActivations[_back]);
			clear(_hiddenStates[_back]);

			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				clear(vl._derivedInput[_back]);
			}
		}

		private:

		static void spStimulus(
			const Image2D &visibleStates,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const bool ignoreMiddle,
			const int2 range)
		{
//#			pragma omp parallel for schedule(dynamic,8)
			for (int x = 0; x < range.x; ++x) {
				const int visiblePositionCenter_x = project(x, hiddenToVisible.x);
				const int fieldLowerBound_x = visiblePositionCenter_x - radius;

#				pragma ivdep
				for (int y = 0; y < range.y; ++y) {
					const int visiblePositionCenter_y = project(y, hiddenToVisible.y);
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					float subSum = 0.0f;
					float stateSum = 0.0f;

#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;

						if (inBounds0(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) { // loop peeling is inefficient 
								//if (ignoreMiddle && (dx == 0) && (dy == 0)) {
									// do nothing;
								//} else {
									const int visiblePosition_y = visiblePositionCenter_y + dy;
									if (inBounds0(visiblePosition_y, visibleSize.y)) {
										const int offset_y = visiblePosition_y - fieldLowerBound_y;

										const int wi = offset_y + (offset_x * ((radius * 2) + 1));
										const float weight = read_3D(weights, x, y, wi);
										const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
										subSum += visibleState * weight;
										stateSum += visibleState;
									}
								//}
							}
						}
					}

					if (ignoreMiddle) { // substract the visible state that corresponds to dx=dy=0
						const int dx = 0;
						const int dy = 0;

						const int visiblePosition_x = visiblePositionCenter_x + dx;
						const int visiblePosition_y = visiblePositionCenter_y + dy;

						const int offset_x = visiblePosition_x - fieldLowerBound_x;
						const int offset_y = visiblePosition_y - fieldLowerBound_y;

						const int wi = offset_y + (offset_x * ((radius * 2) + 1));
						const float weight = read_3D(weights, x, y, wi);
						const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);

						subSum -= visibleState * weight;
						stateSum -= visibleState;
					}

					const float sum = read_2D(hiddenSummationTempBack, x, y);
					write_2D(hiddenSummationTempFront, x, y, sum + subSum / std::max(0.0001f, stateSum));
				}
			}
		}

		static void spActivate(
			const Image2D &stimuli,
			//const Image2D /*&hiddenStates*/,
			const Image2D &biases,
			//const Image2D /*&hiddenActivationsBack*/,
			Image2D &hiddenActivationsFront, // write only
			const int2 range)
		{
			if (true) {
				const int nElements = range.x * range.y;
				for (int i = 0; i < nElements; ++i) {
					const float stimulus = stimuli._data[i];
					const float bias = biases._data[i];
					const float activation = stimulus + bias;
					hiddenActivationsFront._data[i] = activation;
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
			const Image2D &activations,
			Image2D &hiddenStatesFront, // write only
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

#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int otherPosition_x = x + dx;

						if (inBounds0(otherPosition_x, hiddenSize.x)) {
#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) {
								if (dx == 0 && dy == 0) {
									// do nothing
								}
								else {
									const int otherPosition_y = y + dy;
									if (inBounds0(otherPosition_y, hiddenSize.y)) {
										float otherActivation = read_2D(activations, otherPosition_x, otherPosition_y);
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

		static void spLearnWeights(
			const Image2D &hiddenStates,
			const Image2D &visibleStates,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 visibleSize, 
			const float2 hiddenToVisible, 
			const int radius, 
			//const float /*activeRatio*/, //unused
			const float weightAlpha,
			const int2 range)
		{
//#			pragma omp parallel for schedule(dynamic,8)
			for (int x = 0; x < range.x; ++x) {
				const int visiblePositionCenter_x = project(x, hiddenToVisible.x);
				const int fieldLowerBound_x = visiblePositionCenter_x - radius;

#				pragma ivdep 
				for (int y = 0; y < range.y; ++y) {
					const int visiblePositionCenter_y = project(y, hiddenToVisible.y);
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;
					const float hiddenState = read_2D(hiddenStates, x, y);

#					pragma ivdep 
					for (int dx = -radius; dx <= radius; dx++) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;
						if (inBounds0(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

#							pragma ivdep 
							for (int dy = -radius; dy <= radius; dy++) {
								const int visiblePosition_y = visiblePositionCenter_y + dy;
								if (inBounds0(visiblePosition_y, visibleSize.y)) {
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + (offset_x * ((radius * 2) + 1));
									const float weightPrev = read_3D(weightsBack, x, y, wi);
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
									const float learn = hiddenState * (visibleState - weightPrev);

									float weight = weightPrev + weightAlpha * learn;
									weight = (weight > 1.0f) ? 1.0f : ((weight < 0.0f) ? 0.0f : weight);
									
									write_3D(weightsFront, x, y, wi, weight);
								}
							}
						}
					}
				}
			}
		}

		static void spLearnWeights_SpeedTest() {
			printf("Running spLearnWeights_SpeedTest\n");

			const size_t nExperiments = 5000;
			const size_t nRandomNumbers = 10000;
			const unsigned int randSeed = 0xF00FF00F;

			Image2D hiddenStates = Image2D();


			//----------------------------------------------------------------------------------
			double minH1 = std::numeric_limits<double>::max();
			size_t totalSum = 0;
			unsigned int randInt = randSeed;

			for (size_t i = 0; i < nExperiments; ++i) {
				tools::reset_and_start_timer();
				const double dt = ::tools::get_elapsed_mcycles();
				minH1 = std::min(minH1, dt);
			}
			printf("[spLearnWeights Reference]: %2.5f Mcycles, totalSum=%llu\n", minH1, totalSum);

			//----------------------------------------------------------------------------------
			double minH2 = std::numeric_limits<double>::max();
			totalSum = 0;
			randInt = randSeed;
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();
				const double dt = ::tools::get_elapsed_mcycles();
				minH2 = std::min(minH2, dt);
			}
			printf("[spLearnWeights Fast1    ]: %2.5f Mcycles, totalSum=%llu\n", minH2, totalSum);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", minH1 / minH2);
		}



		static void spLearnBiases(
			const Image2D &stimuli,
			//const Image2D /*&hiddenStates*/, // unused
			const Image2D &hiddenBiasesBack,
			Image2D &hiddenBiasesFront, //write only
			//const float /*activeRatio*/, // unused
			const float biasAlpha,
			const int2 range)
		{
			if (true) {
				const int nElements = range.x * range.y;
#				pragma ivdep
				for (int i = 0; i < nElements; ++i) {
					float stimulus = stimuli._data[i];
					float hiddenBiasPrev = hiddenBiasesBack._data[i];
					hiddenBiasesFront._data[i] = hiddenBiasPrev + (biasAlpha * (-stimulus - hiddenBiasPrev));
				}
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {
						float stimulus = read_2D(stimuli, x, y);
						//float hiddenState = read_imagef_2D(hiddenStates, hiddenPosition); //TODO: unused
						float hiddenBiasPrev = read_2D(hiddenBiasesBack, x, y);
						write_2D(hiddenBiasesFront, x, y, hiddenBiasPrev + (biasAlpha * (-stimulus - hiddenBiasPrev)));
					}
				}
			}
		}

		static void spDeriveInputs(
			const Image2D &inputs,
			//const Image2D /*&outputsBack*/, // unused
			Image2D &outputsFront, // write only
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

	namespace speedtest {

	}

}

