// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.ipp"

namespace feynman {

	// Sparse coder. Learns a spatial-only sparse code
	class SparseCoder {
	public:

		const static bool INFO = true;

		//Visible layer descriptor
		struct VisibleLayerDesc {

			//Size of layer
			int2 _size;

			//radius onto input
			int _radius;

			//Whether or not the middle (center) input should be ignored (self in recurrent schemes)
			bool _ignoreMiddle;

			//Learning rate
			float _weightAlpha;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 8, 8 }),
				_radius(6),
				_ignoreMiddle(false),
				_weightAlpha(0.02f)
			{}
		};

		//Visible layer
		struct VisibleLayer {

			//Possibly manipulated input
			DoubleBuffer2D<float> _derivedInput;

			//Temporary buffer for reconstruction error
			Image2D<float> _reconError;

			//Weights
			DoubleBuffer3D<float> _weights; // Encoding weights (creates spatio-temporal sparse code)

			//Transformations
			float2 _hiddenToVisible;
			float2 _visibleToHidden;
			int2 _reverseRadii;
		};

	private:

		//Hidden states, thresholds (similar to biases)
		DoubleBuffer2D<float> _hiddenStates;
		DoubleBuffer2D<float> _hiddenThresholds;

		//Hidden size
		int2 _hiddenSize;

		//Inhibition radius
		int _inhibitionRadius = 0;

		//Hidden stimulus summation temporary buffer
		DoubleBuffer2D<float> _hiddenStimulusSummationTemp;

		//Layers and descs
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization.
		Requires initialization information.
		\param visibleLayerDescs descriptors for all input layers.
		\param hiddenSize hidden (output) size (2D).
		\param inhibitionRadius inhibitory radius.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param initThresholdRange are the minimum and maximum range values for threshold initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const int2 hiddenSize,
			const int inhibitionRadius,
			const float2 initWeightRange,
			const float2 initThresholdRange,
			std::mt19937 &rng)
		{
			_visibleLayerDescs = visibleLayerDescs;
			_hiddenSize = hiddenSize;
			_inhibitionRadius = inhibitionRadius;

			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
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
					const int weightDiam = vld._radius * 2 + 1;
					const int numWeights = weightDiam * weightDiam;
					const int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._weights = createDoubleBuffer3D<float>(weightsSize);
					randomUniform3D(vl._weights[_back], weightsSize, initWeightRange, rng);
				}

				vl._derivedInput = createDoubleBuffer2D<float>(vld._size);
				clear(vl._derivedInput[_back]);
				vl._reconError = Image2D<float>(vld._size);
			}

			// Hidden state data
			_hiddenStates = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenThresholds = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenStimulusSummationTemp = createDoubleBuffer2D<float>(_hiddenSize);
			clear(_hiddenStates[_back]);

			randomUniform2D(_hiddenThresholds[_back], _hiddenSize, initThresholdRange, rng);
		}

		/*!
		\brief Activate predictor
		\param visibleStates input layer states.
		\param inputTraceDecay decay of input averaging trace.
		\param activeRatio % active units.
		\param rng a random number generator.
		*/
		void activate(
			const std::vector<Image2D<float>> &visibleStates,
			const float /*inputTraceDecay*/, // unused
			const float activeRatio,
			std::mt19937 /*&rng*/)	// unused
		{
			//if (INFO) printf("INFO: SparseCoder::activate: nVisibleLayer=%llu\n", _visibleLayers.size());

			// Derive inputs
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				//if (INFO) printf("INFO: SparseCoder::activate: scDeriveInputs-level %llu\n", vli);
				//plots::plotImage(visibleStates[vli], 16.0f, false, "SparseCoder:visibleStates:");

				// copy all visible states (unaltered) to derivedInputs
				scDeriveInputs(
					visibleStates[vli],			// in
					//vl._derivedInput[_back],	// unused
					vl._derivedInput[_front],	// out
					//inputTraceDecay,
					vld._size);

				//plots::plotImage(vl._derivedInput[_front], 16.0f, false, "SparseCoder:derivedInput:");
			}

			// Start by clearing stimulus summation buffer to biases
			copy(_hiddenThresholds[_back], _hiddenStimulusSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				//if (INFO) printf("INFO: SparseCoder::activate: scStimulus-level %llu\n", vli);
				//plots::plotImage(_hiddenStimulusSummationTemp[_back], 16.0f, "SparseCoder:_hiddenStimulusSummationTemp: in");

				scStimulus(
					vl._derivedInput[_front],				// in
					_hiddenStimulusSummationTemp[_back],	// in
					_hiddenStimulusSummationTemp[_front],	// out
					vl._weights[_back],						// in
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					vld._ignoreMiddle,
					_hiddenSize);
				//plots::plotImage(_hiddenStimulusSummationTemp[_front], 8.0f, true, "SparseCoder:_hiddenStimulusSummationTemp: out");

				std::swap(_hiddenStimulusSummationTemp[_front], _hiddenStimulusSummationTemp[_back]);
			}
			//plots::plotImage(_hiddenStimulusSummationTemp[_back], 8.0f, true, "SparseCoder:_hiddenStimulusSummationTemp:");

			// Solve hidden
			scSolveHidden(
				_hiddenStimulusSummationTemp[_back],	// in
				_hiddenStates[_front],					// out
				_hiddenSize,
				_inhibitionRadius,
				activeRatio,
				_hiddenSize);
			//plots::plotImage(_hiddenStates[_front], 8.0f, true, "SparseCoder:_hiddenStates:");
		}

		//End a simulation step
		void stepEnd() {
			std::swap(_hiddenStates[_front], _hiddenStates[_back]);
			// Swap buffers
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
			}
		}

		/*!
		\brief Learning
		\param thresholdAlpha threshold learning rate.
		\param activeRatio % active units.
		*/
		void learn(
			const float thresholdAlpha,
			const float activeRatio)
		{
			// Reverse
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				scReverse(
					_hiddenStates[_front],		// in
					vl._derivedInput[_front],	// in
					vl._reconError,				// out
					vl._weights[_back],			// in
					vld._size,
					//_hiddenSize,
					vl._visibleToHidden,
					vl._hiddenToVisible,
					vld._radius,
					vl._reverseRadii,
					vld._size);

				//plots::plotImage(vl._reconError, 16.0f, "reconError");
			}

			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				scLearnWeights(
					_hiddenStates[_front],		// in
					//_hiddenStates[_back],		// unused
					vl._reconError,				// in
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
			scLearnThresholds(
				_hiddenStates[_front],			// in
				_hiddenThresholds[_back],		// in
				_hiddenThresholds[_front],		// out
				thresholdAlpha,
				activeRatio,
				_hiddenSize);

			//plots::plotImage(_hiddenThresholds[_front], 16.0f, "_hiddenThresholds", true);

			std::swap(_hiddenThresholds[_front], _hiddenThresholds[_back]);
		}

		//Reconstruct image from an SDR
		void reconstruct(
			const Image2D<float> &hiddenStates,
			std::vector<Image2D<float>> &reconstructions)
		{
			if (reconstructions.size() != _visibleLayers.size()) {
				printf("WARNING: incorrect number of reconstructions");
				return;
			}
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				scReconstruct(
					hiddenStates,
					reconstructions[vli],
					vl._weights[_back],
					vld._size,
					_hiddenSize,
					vl._visibleToHidden,
					vl._hiddenToVisible,
					vld._radius,
					vl._reverseRadii,
					vld._size);
			}
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
		const DoubleBuffer2D<float> &getHiddenStates() const {
			return _hiddenStates;
		}

		// Get hidden biases
		const DoubleBuffer2D<float> &getHiddenThresholds() const {
			return _hiddenThresholds;
		}

		//Clear the working memory
		void clearMemory()
		{
			clear(_hiddenStates[_back]);
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				clear(vl._derivedInput[_back]);
			}
		}

	private:

		/**
		Writes hiddenSummationTempFront
		*/
		static void scStimulus(
			const Image2D<float> &visibleStates, // was float2
			const Image2D<float> &hiddenSummationTempBack,
			Image2D<float> &hiddenSummationTempFront, // write only
			const Image3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const bool ignoreMiddle,
			const int2 range)
		{
#			pragma ivdep
			for (int x = 0; x < range.x; ++x) {
				const int visiblePositionCenter_x = static_cast<int>(x * hiddenToVisible.x + 0.5f);
				const int fieldLowerBound_x = visiblePositionCenter_x - radius;

#				pragma ivdep
				for (int y = 0; y < range.y; ++y) {
					const int visiblePositionCenter_y = static_cast<int>(y * hiddenToVisible.y + 0.5f);
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					float subSum = 0.0f;	// subSum is the sum of the simulus that hidden node (x,y) receives from 
					int count = 0;

					//if ((x == 10) && (y == 10)) 
					//	printf("BREAK");

#					pragma ivdep
					for (int dx = -radius; dx <= radius; dx++) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;

						if (inBounds0(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

#							pragma ivdep
							for (int dy = -radius; dy <= radius; dy++) {
								if (ignoreMiddle && (dx == 0) && (dy == 0))
									continue;

								const int visiblePosition_y = visiblePositionCenter_y + dy;

								if (inBounds0(visiblePosition_y, visibleSize.y)) {
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + (offset_x * ((radius * 2) + 1));
									const float weight = read_3D(weights, x, y, wi);
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);// .x;
									subSum += visibleState * weight;
									count++;
								}
							}
						}
					}

					const float sum = read_2D(hiddenSummationTempBack, x, y);
					const float hiddenSummation = sum + subSum / std::max(1, count);
					//printf("SparseCoder:scStimulus: pos(%i,%i): sum=%f; subSum=%f\n", x, y, sum, subSum);
					write_2D(hiddenSummationTempFront, x, y, hiddenSummation);
				}
			}
		}

		static void scReverse(
			const Image2D<float> &hiddenStates,
			const Image2D<float> &visibleStates, // was float2
			Image2D<float> &reconErrors, // write only
			const Image3D<float> &weights,
			//const int2 /*visibleSize*/, // unused
			const int2 hiddenSize,
			const float2 visibleToHidden,
			const float2 hiddenToVisible,
			const int radius,
			const int2 reverseRadii,
			const int2 range)
		{
#			pragma ivdep
			for (int x = 0; x < range.x; ++x) {
				const int hiddenPositionCenter_x = static_cast<int>(x * visibleToHidden.x + 0.5f);

#				pragma ivdep
				for (int y = 0; y < range.y; ++y) {
					const int hiddenPositionCenter_y = static_cast<int>(y * visibleToHidden.y + 0.5f);

					float recon = 0.0f;
					//float div = 0.0f;

#					pragma ivdep
					for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++) {
						const int hiddenPosition_x = hiddenPositionCenter_x + dx;

						if (inBounds0(hiddenPosition_x, hiddenSize.x)) {
							const int fieldCenter_x = static_cast<int>(hiddenPosition_x * hiddenToVisible.x + 0.5f);
							const int fieldLowerBound_x = fieldCenter_x - radius;
							const int fieldUpperBound_x = fieldCenter_x + radius + 1; // So is included in inBounds

							// Check for containment
							if (inBounds(x, fieldLowerBound_x, fieldUpperBound_x)) {
								const int offset_x = x - fieldLowerBound_x;

#								pragma ivdep
								for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
									const int hiddenPosition_y = hiddenPositionCenter_y + dy;

									if (inBounds0(hiddenPosition_y, hiddenSize.y)) {
										// Next layer node's receptive field
										const int fieldCenter_y = static_cast<int>(hiddenPosition_y * hiddenToVisible.y + 0.5f);

										const int fieldLowerBound_y = fieldCenter_y - radius;
										const int fieldUpperBound_y = fieldCenter_y + radius + 1; // So is included in inBounds

										// Check for containment
										if (inBounds(y, fieldLowerBound_y, fieldUpperBound_y)) {
											const int offset_y = y - fieldLowerBound_y;

											const float hiddenState = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
											const int wi = offset_y + (offset_x * ((radius * 2) + 1));
											const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
											recon += hiddenState * weight;
											//div += hiddenState;
										}
									}
								}
							}
						}
					}
					const float visibleState = read_2D(visibleStates, x, y);// .x;
					write_2D(reconErrors, x, y, visibleState - recon);
				}
			}
		}

		//Create SDR in hiddenStatesFront from activations
		static void scSolveHidden(
			const Image2D<float> &activations,
			Image2D<float> &hiddenStatesFront, // write only
			const int2 hiddenSize,
			const int radius,
			const float activeRatio,
			const int2 range)
		{
			clear(hiddenStatesFront);

#			pragma ivdep
			for (int x = 0; x < range.x; ++x) {
#				pragma ivdep
				for (int y = 0; y < range.y; ++y) {

					//if (x == 10 && y == 10)
					//	printf("BREAK");

					// for every pos (x,y) in activations, count the neighbours (in range) that 
					// have an activation that is higher (or equal)

					const float activation = read_2D(activations, x, y);
					int inhibition = 0;
					int count = 0;

#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int otherPosition_x = x + dx;

						if (inBounds0(otherPosition_x, hiddenSize.x)) {

#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) {
								if (dx == 0 && dy == 0)
									continue;

								const int otherPosition_y = y + dy;

								if (inBounds0(otherPosition_y, hiddenSize.y)) {
									const float otherActivation = read_2D(activations, otherPosition_x, otherPosition_y);
									inhibition += (otherActivation >= activation) ? 1 : 0;
									count++;
								}
							}
						}
					}

					// if the number of neighbours (with higher activation) is smaller than activeRatio, 
					// then pos (x,y) is active;
					if (inhibition <= (activeRatio * count)) {
						write_2D(hiddenStatesFront, x, y, 1.0f);
					}
				}
			}
		}

		static void scLearnWeights(
			const Image2D<float> &hiddenStates,
			//const Image2D<float> &hiddenStatesPrev, // unused
			const Image2D<float> &reconErrors,
			const Image3D<float> &weightsBack,
			Image3D<float> &weightsFront, // write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			//const float /*activeRatio*/,
			const float weightAlpha,
			const int2 range)
		{
			for (int x = 0; x < range.x; ++x) {
				const int visiblePositionCenter_x = static_cast<int>(x * hiddenToVisible.x + 0.5f);
				const int fieldLowerBound_x = visiblePositionCenter_x - radius;

				for (int y = 0; y < range.y; ++y) {
					const int visiblePositionCenter_y = static_cast<int>(y * hiddenToVisible.y + 0.5f);
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					const float hiddenState = read_2D(hiddenStates, x, y);
					//const float hiddenStatePrev = read_imagef_2D(hiddenStatesPrev, hiddenPosition); // unused

#					pragma ivdep
					for (int dx = -radius; dx <= radius; ++dx) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;

						if (inBounds0(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

#							pragma ivdep
							for (int dy = -radius; dy <= radius; ++dy) {
								const int visiblePosition_y = visiblePositionCenter_y + dy;

								if (inBounds0(visiblePosition_y, visibleSize.y)) {
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + (offset_x * ((radius * 2) + 1));
									const float weightPrev = read_3D(weightsBack, x, y, wi);
									const float reconError = read_2D(reconErrors, visiblePosition_x, visiblePosition_y);
									const float newWeight = weightPrev + weightAlpha * hiddenState * reconError;
									write_3D(weightsFront, x, y, wi, newWeight);
								}
							}
						}
					}
				}
			}
		}

		static void scLearnThresholds(
			const Image2D<float> &hiddenStates,
			const Image2D<float> &hiddenThresholdsBack,
			Image2D<float> &hiddenThresholdsFront, // write only
			const float thresholdAlpha,
			const float activeRatio,
			const int2 range)
		{
			if (true) {
				const int nElements = range.x * range.y;
#				pragma ivdep
				for (int i = 0; i < nElements; ++i) {
					const float hiddenState = hiddenStates._data[i];
					const float hiddenThresholdPrev = hiddenThresholdsBack._data[i];
					hiddenThresholdsFront._data[i] = hiddenThresholdPrev + thresholdAlpha * (activeRatio - hiddenState);
				}
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {
						const float hiddenState = read_2D(hiddenStates, x, y);
						const float hiddenThresholdPrev = read_2D(hiddenThresholdsBack, x, y);
						write_2D(hiddenThresholdsFront, x, y, hiddenThresholdPrev + thresholdAlpha * (activeRatio - hiddenState));
					}
				}
			}
		}

		/**
		Update all positions in outputsFront with:
		1] unaltered input from inputs,
		2] exponential moving average of current output and the previous output
		*/
		static void scDeriveInputs(
			const Image2D<float> &inputs,
			//const Image2D<float> /*&outputsBack*/, // unused: was float2
			Image2D<float> &outputsFront, // write only: was float2
			//const float /*lambda*/, // unused
			const int2 range)
		{
			//if (INFO) printf("INFO: SparseCoder::scDeriveInputs: lambda=%f\n", lambda);
			if (true) {
				copy(inputs, outputsFront);
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {
						const float input = read_2D(inputs, x, y);
						//const float outputPrev = read_2D(outputsBack, x, y).y;
						//const float outputNew = (lambda * outputPrev) + ((1.0f - lambda) * input);
						//printf("SparseCoder:scDeriveInputs: pos(%i,%i): input=%f; outputPrev=%f; outputNew=%f\n", x, y, input, outputPrev, outputNew);
						write_2D(outputsFront, x, y, input);// float2{ input, outputNew });
					}
				}
			}
		}

		static void scReconstruct(
			const Image2D<float> &hiddenStates,
			Image2D<float> &reconstruction, // write only
			const Image3D<float> &weights,
			const int2 /*visibleSize*/,
			const int2 hiddenSize,
			const float2 visibleToHidden,
			const float2 hiddenToVisible,
			const int radius,
			const int2 reverseRadii,
			const int2 range)
		{
#			pragma ivdep
			for (int x = 0; x < range.x; ++x) {
				const int hiddenPositionCenter_x = static_cast<int>(x * visibleToHidden.x + 0.5f);
				
#				pragma ivdep
				for (int y = 0; y < range.y; ++y) {
					const int hiddenPositionCenter_y = static_cast<int>(y * visibleToHidden.y + 0.5f);

					float recon = 0.0f;
					//float div = 0.0f;

#					pragma ivdep
					for (int dx = -reverseRadii.x; dx <= reverseRadii.x; ++dx) {
						const int hiddenPosition_x = hiddenPositionCenter_x + dx;

						if (inBounds0(hiddenPosition_x, hiddenSize.x)) {
							const int fieldCenter_x = static_cast<int>(hiddenPosition_x * hiddenToVisible.x + 0.5f);
							const int fieldLowerBound_x = fieldCenter_x - radius;
							const int fieldUpperBound_x = fieldCenter_x + radius + 1; // So is included in inBounds
							const int offset_x = x - fieldLowerBound_x;

#							pragma ivdep
							for (int dy = -reverseRadii.y; dy <= reverseRadii.y; ++dy) {
								const int hiddenPosition_y = hiddenPositionCenter_y + dy;

								if (inBounds0(hiddenPosition_y, hiddenSize.y)) {
									// Next layer node's receptive field
									const int fieldCenter_y = static_cast<int>(hiddenPosition_y * hiddenToVisible.y + 0.5f);
									const int fieldLowerBound_y = fieldCenter_y - radius;
									const int fieldUpperBound_y = fieldCenter_y + radius + 1; // So is included in inBounds

									// Check for containment
									if (inBounds(x, fieldLowerBound_x, fieldUpperBound_x) && inBounds(y, fieldLowerBound_y, fieldUpperBound_y)) {
										const int offset_y = y - fieldLowerBound_y;
										const float hiddenState = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
										const int wi = offset_y + (offset_x * ((radius * 2) + 1));
										const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
										recon += hiddenState * weight;
										//div += hiddenState;  //TODO never read
									}
								}
							}
						}
					}
					write_2D(reconstruction, x, y, recon);
				}
			}
		}
	};
}