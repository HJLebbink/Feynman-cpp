// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>		// for cerr and cout

#include "Helpers.ipp"

namespace feynman {

	//Predictor layer. A 2D perceptron decoder (Predictor) layer.
	class PredictorLayer {
	public:

		//Layer descriptor
		struct VisibleLayerDesc {

			//Layer properties: Input size, radius onto input, learning rate.
			int2 _size;
			int _radius;
			float _alpha;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 16, 16 }),
				_radius(10),
				_alpha(0.01f)
			{}
		};

		//Layer
		struct VisibleLayer {

			//Layer parameters
			DoubleBuffer3D<float> _weights;

			float2 _hiddenToVisible;
			float2 _visibleToHidden;
			int2 _reverseRadii;
		};

	private:

		//Size of the prediction
		int2 _hiddenSize;

		//Hidden stimulus summation temporary buffer
		DoubleBuffer2D<float> _hiddenSummationTemp;

		//Predictions
		DoubleBuffer2D<float> _hiddenStates;

		//Layers and descs
		std::vector<VisibleLayer> _visibleLayers;
		std::vector<VisibleLayerDesc> _visibleLayerDescs;

	public:

		/*!
		\brief Create a predictor layer with random initialization.
		Requires initialization information.
		\param hiddenSize size of the predictions (output).
		\param visibleLayerDescs are descriptors for visible layers.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const int2 hiddenSize,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			_visibleLayerDescs = visibleLayerDescs;
			_hiddenSize = hiddenSize;
			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

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
			}
			// Hidden state data
			_hiddenStates = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenSummationTemp = createDoubleBuffer2D<float>(_hiddenSize);

			clear(_hiddenStates[_back]);
		}

		/*!
		\brief Activate predictor (predict values)
		\param visibleStates the input layer states.
		\param threshold whether or not the output should be thresholded (binary).
		*/
		void activate(
			const std::vector<Image2D<float>> &visibleStates,
			const bool threshold)
		{
			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				plStimulus(
					visibleStates[vli],				// in
					_hiddenSummationTemp[_back],	// in
					_hiddenSummationTemp[_front],	// out
					vl._weights[_back],				// in
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					_hiddenSize);

				//plots::plotImage(_hiddenSummationTemp[_front], 4.0f, false, "PredictionLayer:_hiddenSummationTemp:");

				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}
			if (threshold) {
				plThreshold(
					_hiddenSummationTemp[_back],	// in
					_hiddenStates[_front],			// out
					_hiddenSize);
			}
			else {
				// Copy to hidden states
				copy(_hiddenSummationTemp[_back], _hiddenStates[_front]);
			}
		}

		/*!
		\brief Learn predictor
		\param targets target values to update towards.
		\param visibleStatesPrev the input states of the !previous! timestep.
		*/
		void learn(
			const Image2D<float> &targets,
			const std::vector<Image2D<float>> &visibleStatesPrev)
		{
			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				plLearnPredWeights(
					visibleStatesPrev[vli],
					targets,
					_hiddenStates[_back],
					vl._weights[_back],
					vl._weights[_front],
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					vld._alpha, 
					_hiddenSize);

				std::swap(vl._weights[_front], vl._weights[_back]);
			}
		}

		//Step end (buffer swap)
		void stepEnd() {
			std::swap(_hiddenStates[_front], _hiddenStates[_back]);
		}

		//Clear memory (recurrent data)
		void clearMemory() {
			clear(_hiddenStates[_back]);
		}

		//Get number of layers
		size_t getNumLayers() const {
			return _visibleLayers.size();
		}

		//Get access to a layer
		const VisibleLayer &getLayer(int index) const {
			return _visibleLayers[index];
		}

		//Get access to a layer descriptor
		const VisibleLayerDesc &getLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		//Get the predictions
		const DoubleBuffer2D<float> &getHiddenStates() const {
			return _hiddenStates;
		}

		//Get the hidden size
		int2 getHiddenSize() const {
			return _hiddenSize;
		}

	private:

		static void plStimulus(
			const Image2D<float> &visibleStates,
			const Image2D<float> &hiddenSummationTempBack,
			Image2D<float> &hiddenSummationTempFront, // write only
			const Image3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
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
									const float weight = read_3D(weights, x, y, wi);
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
									subSum += visibleState * weight;
								}
							}
						}
					}

					const float sum = read_2D(hiddenSummationTempBack, x, y);
					write_2D(hiddenSummationTempFront, x, y, sum + subSum);
				}
			}
		}

		static void plThreshold(
			const Image2D<float> &stimuli,
			Image2D<float> &thresholded, // write only
			const int2 range)
		{
			if (true) {
				const int nElements = range.x * range.y;
#				pragma ivdep
				for (int i = 0; i < nElements; ++i) {
					const float stimulus = stimuli._data[i];
					thresholded._data[i] = (stimulus > 0.5f) ? 1.0f : 0.0f;
				}
			}
			else {
#				pragma ivdep
				for (int x = 0; x < range.x; ++x) {
#					pragma ivdep
					for (int y = 0; y < range.y; ++y) {
						const float stimulus = read_2D(stimuli, x, y);
						write_2D(thresholded, x, y, (stimulus > 0.5f) ? 1.0f : 0.0f);
					}
				}
			}
		}

		static void plLearnPredWeights(
			const Image2D<float> &visibleStatesPrev,
			const Image2D<float> &targets,
			const Image2D<float> &hiddenStatesPrev,
			const Image3D<float> &weightsBack,
			Image3D<float> &weightsFront, //write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const float alpha,
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
					const float error = read_2D(targets, x ,y) - read_2D(hiddenStatesPrev, x, y);

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
									const float visibleStatePrev = read_2D(visibleStatesPrev, visiblePosition_x, visiblePosition_y);
									const float weight = weightPrev + (alpha * error * visibleStatePrev);
									write_3D(weightsFront, x, y, wi, weight);
								}
							}
						}
					}
				}
			}
		}
	};
}
