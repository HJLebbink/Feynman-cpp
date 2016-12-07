// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>		// for cerr and cout
#include <memory>
#include <algorithm>

#include "Helpers.ipp"
#include "FixedPoint.ipp"
#include "SparseFeatures.ipp"
#include "timing.h"
#include "PlotDebug.ipp"

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
				_radius(8),
				_alpha(0.04f)
			{}
		};

		//Layer
		struct VisibleLayer {

			//Layer parameters
			DoubleBuffer2D<float> _derivedInput;
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

		//Encoder corresponding to this decoder, if available (else nullptr)
		std::shared_ptr<SparseFeatures> _inhibitSparseFeatures;

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
			const std::shared_ptr<SparseFeatures> &inhibitSparseFeatures,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			// last checked: 28-nov 2016

			_visibleLayerDescs = visibleLayerDescs;
			_hiddenSize = hiddenSize;
			_inhibitSparseFeatures = inhibitSparseFeatures;
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
			const std::vector<Array2D<float>> &visibleStates,
			std::mt19937 &rng)
		{
			// last checked: 28-nov 2016

			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				//plots::plotImage(visibleStates[vli], 6, "PredictorLayer:activate:visibleStates" + std::to_string(vli));

				// Derive inputs
				if (EXPLAIN) std::cout << "EXPLAIN: PredictorLayer:activate: visible layer " << vli << "/" << _visibleLayers.size() << ": deriving inputs." << std::endl;
				plDeriveInputs(
					visibleStates[vli],				// in
					vl._derivedInput[_back],		// in
					vl._derivedInput[_front],		// out: note: _derivedInput are used in learn
					vld._size
				);

				//plots::plotImage(vl._derivedInput[_front], 6, "PredictorLayer:activate:derivedInput" + std::to_string(vli));

				if (EXPLAIN) std::cout << "EXPLAIN: PredictorLayer:activate: visible layer " << vli << "/" << _visibleLayers.size() << ": adding inputs to stimuls influx." << std::endl;
				plStimulus(
					vl._derivedInput[_front],		// in
					_hiddenSummationTemp[_back],	// in
					_hiddenSummationTemp[_front],	// out
					vl._weights[_back],				// in
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					_hiddenSize
				);

				//plots::plotImage(_hiddenSummationTemp[_front], 8, "PredictorLayer:activate:hiddenSummationTemp" + std::to_string(vli));

				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}

			//plots::plotImage(_hiddenSummationTemp[_back], 8, "PredictorLayer:activate:hiddenSummationTemp");

			if (_inhibitSparseFeatures != nullptr) {
				if (EXPLAIN) std::cout << "EXPLAIN: PredictorLayer:activate: calculating hidden state SDR based on stimuls influx." << std::endl;
				_inhibitSparseFeatures->inhibit(_hiddenSummationTemp[_back], _hiddenStates[_front], rng);
			} else {
				if (EXPLAIN) std::cout << "EXPLAIN: PredictorLayer:activate: hidden state is equal to stimuls influx." << std::endl;
				copy(_hiddenSummationTemp[_back], _hiddenStates[_front]);
			}
			//plots::plotImage(_hiddenStates[_front], 6, "PredictionLayer:activate:hiddenStates");
		}

		/*!
		\brief Learn predictor
		\param targets target values to update towards.
		*/
		void learn(
			const Array2D<float> &targets)
		{
			// last checked: 28-nov 2016

			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				//plots::plotImage(_hiddenStates[_back], 8, "PredictorLayer:learn:hiddenState" + std::to_string(vli));

				if (EXPLAIN) std::cout << "EXPLAIN: PredictorLayer:learn: layer " << vli << "/" << _visibleLayers.size() << ": updating weights based on error between targets and derived inputs." << std::endl;
				plLearnPredWeights(
					vl._derivedInput[_back],// in
					targets,				// in
					_hiddenStates[_back],	// in
					vl._weights[_back],		// in
					vl._weights[_front],	// out
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
			// last checked: 25-nov 2016
			std::swap(_hiddenStates[_front], _hiddenStates[_back]);

			// Swap buffers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				//VisibleLayerDesc &vld = _visibleLayerDescs[vli];
				std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
			}
		}

		//Clear memory (recurrent data)
		void clearMemory() {
			// last checked: 25-nov 2016
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

		static void speedTest(size_t nExperiments = 1)
		{
#			ifdef _DEBUG
			nExperiments = 1;
#			endif
			speedTest_plLearnPredWeights(nExperiments);
			speedTest_plStimulus(nExperiments);
		}

		static void speedTest_plLearnPredWeights(const size_t nExperiments = 1) {
			printf("Running PredictorLayer::speedTest\n");
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

			const int RADIUS = 6;
			const float weightAlpha = 0.002;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Array2D<float> visibleStatesPrev = Array2D<float>(visibleSize);
			Array2D<float> targets = Array2D<float>(hiddenSize);
			Array2D<float> hiddenStatesPrev = Array2D<float>(hiddenSize);
			Array3D<float> weightsBack = Array3D<float>(weightsSize);
			Array3D<float> weightsFront0 = Array3D<float>(weightsSize);
			Array3D<float> weightsFront1 = Array3D<float>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { -0.001f, 0.001f };
			randomUniform2D(visibleStatesPrev, initRange, generator);
			randomUniform2D(targets, initRange, generator);
			randomUniform2D(hiddenStatesPrev, initRange, generator);
			randomUniform3D(weightsBack, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				plLearnPredWeights_v0<RADIUS>(
					visibleStatesPrev,		// in
					targets,				// in
					hiddenStatesPrev,		// in
					weightsBack,			// in
					weightsFront0,			// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,			// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[plLearnPredWeights_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				plLearnPredWeights_v1<RADIUS>(
					visibleStatesPrev,		// in
					targets,				// in
					hiddenStatesPrev,		// in
					weightsBack,			// in
					weightsFront1,			// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,			// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[plLearnPredWeights_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < weightsFront1._size.x; ++x) {
				for (int y = 0; y < weightsFront1._size.y; ++y) {
					for (int z = 0; z < weightsFront1._size.z; ++z) {
						const float f0 = read_3D(weightsFront0, x, y, z);
						const float f1 = read_3D(weightsFront1, x, y, z);
						if (f0 != f1) printf("WARNING: PredictorLayer::speedTest_plLearnPredWeights: coord=(%i,%i,%i): f0=%f; f1=%f\n", x, y, z, f0, f1);
					}
				}
			}
		}
		static void speedTest_plStimulus(const size_t nExperiments = 1) {
			printf("Running PredictorLayer::speedTest\n");
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

			const int RADIUS = 6;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Array2D<float> visibleStates = Array2D<float>(visibleSize);
			Array2D<float> hiddenSummationTempBack = Array2D<float>(hiddenSize);
			Array2D<float> hiddenSummationTempFront = Array2D<float>(hiddenSize);
			Array3D<float> weights0 = Array3D<float>(weightsSize);
			Array3D<float> weights1 = Array3D<float>(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 0.001f };
			randomUniform2D(visibleStates, initRange, generator);
			randomUniform2D(hiddenSummationTempBack, initRange, generator);
			randomUniform2D(hiddenSummationTempFront, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				plStimulus_v0<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront,	// in
					weights0,					// out
					visibleSize,
					hiddenToVisible);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[plStimulus_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				plStimulus_v1<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront,	// in
					weights1,					// out
					visibleSize,
					hiddenToVisible);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[plStimulus_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < weights1._size.x; ++x) {
				for (int y = 0; y < weights1._size.y; ++y) {
					for (int z = 0; z < weights1._size.z; ++z) {
						const float f0 = read_3D(weights0, x, y, z);
						const float f1 = read_3D(weights1, x, y, z);
						if (f0 != f1) printf("WARNING: PredictorLayer::speedTest_plStimulus: coord=(%i,%i,%i): f0=%f; f1=%f\n", x, y, z, f0, f1);
					}
				}
			}
		}

	private:

		static void plDeriveInputs(
			const Array2D<float> &inputs,
			const Array2D<float> &outputsBack,
			Array2D<float> &outputsFront,			// write only
			const int2 range)
		{
			// last checked: 24-nov 2016
			copy(inputs, outputsFront);
		}

		template <bool CORNER, int RADIUS>
		static void plStimulus_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible)
		{
			throw 1;

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
				}
			}
			const float sumCurrent = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
			float sumNew = sumCurrent + subSum;
			write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, sumNew);
		}

		template <int RADIUS>
		static void plStimulus_v0(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible)
		{
			throw 1;

			for (int hiddenPosition_x = 0; hiddenPosition_x < hiddenSummationTempBack._size.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < hiddenSummationTempBack._size.y; ++hiddenPosition_y) {
					plStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
			}
		}

		template <int RADIUS>
		static void plStimulus_v1(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible)
		{
			throw 1;
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
					plStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					plStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					plStimulus_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					plStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					plStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible);
				}
			}
		}

		static void plStimulus(
			const Array2D<float> &visibleStates,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const int2 range)
		{
			// last checked: 28-nov 2016
			/*
			switch (radius) {
			case 6: plStimulus_v0<6>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible); break;
			case 8: plStimulus_v0<8>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible); break;
			case 20: plStimulus_v0<20>(visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible); break;
			default: printf("ERROR: SparseFeatures::plStimulus: provided radius %i is not implemented\n", radius); break;
			}
			*/
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
				const int fieldLowerBound_x = visiblePositionCenter_x - radius;

				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {
					const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					float subSum = 0.0f;

					for (int dx = -radius; dx <= radius; ++dx) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;
						if (inBounds(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

							for (int dy = -radius; dy <= radius; ++dy) {
								const int visiblePosition_y = visiblePositionCenter_y + dy;

								if (inBounds(visiblePosition_y, visibleSize.y)) {
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + offset_x * (radius * 2 + 1);
									const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
									const float visibleState = read_2D(visibleStates, visiblePosition_x, visiblePosition_y);
									subSum += visibleState * weight;
								}
							}
						}
					}
					const float sum = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
					const float newValue = sum + subSum;
					write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newValue);
				}
			}
		}

		static void plThreshold(
			const Array2D<float> &stimuli,
			Array2D<float> &thresholded, // write only
			const int2 range)
		{
			// last checked: 25-nov 2016
			const int nElements = range.x * range.y;
#			pragma ivdep
			for (int i = 0; i < nElements; ++i) {
				const float stimulus = stimuli._data_float[i];
				thresholded._data_float[i] = (stimulus > 0.5f) ? 1.0f : 0.0f;
			}
		}

		template <bool CORNER, int RADIUS>
		static void plLearnPredWeights_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const float2 hiddenToVisible,
			const float weightAlpha,
			const Array2D<float> &visibleStatesPrev,
			const Array2D<float> &targets,
			const Array2D<float> &hiddenStatesPrev,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront) //write only
		{
			throw 1;
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleStatesPrev._size.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleStatesPrev._size.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

#			ifdef USE_FIXED_POINT
				FixedP weightAlphaFP = toFixedP(weightAlpha);
				if (weightAlphaFP == 0) printf("WARNING: plLearnPredWeights_kernel weightAlpha=%24.22f (%i) is too small\n", weightAlpha, toFixedP(weightAlpha));

				const float error_old = weightAlpha * (read_2D(targets, hiddenPosition_x, hiddenPosition_y) - read_2D(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y));
				const FixedP2 error = multiply(weightAlphaFP, (read_2D_fixp(targets, hiddenPosition_x, hiddenPosition_y) - read_2D_fixp(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y)));

				//printf("INFO: plLearnPredWeights_kernel weightAlpha=%24.22f (%i); error=%24.22 (%i); error_old=%24.22f (%i)\n", weightAlpha, toFixedP(weightAlpha), toFloat(error), error, error_old, toFixedP(error_old));

#				pragma ivdep
				for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
					const int offset_x = visiblePosition_x - fieldLowerBound_x;

#					pragma ivdep
					for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
						const int offset_y = visiblePosition_y - fieldLowerBound_y;

						const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
						const FixedP weightPrev = read_3D_fixp(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
						const FixedP visibleStatePrev = read_2D_fixp(visibleStatesPrev, visiblePosition_x, visiblePosition_y);
						const FixedP wD = multiply_saturate(error, visibleStatePrev);
						const FixedP weight = add_saturate(weightPrev, wD);

						//printf("WARNING: plLearnPredWeights_kernel weightPrev=%24.22f (%i); weight=%24.22f (%i)\n", toFloat(weightPrev), weightPrev, toFloat(weight), weight);
						write_3D_fixp(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weight);
					}
				}
#			else
				const float error = weightAlpha * (read_2D(targets, hiddenPosition_x, hiddenPosition_y) - read_2D(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y));

#				pragma ivdep
				for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
					const int offset_x = visiblePosition_x - fieldLowerBound_x;

#					pragma ivdep
					for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
						const int offset_y = visiblePosition_y - fieldLowerBound_y;

						const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
						const float weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
						const float visibleStatePrev = read_2D(visibleStatesPrev, visiblePosition_x, visiblePosition_y);
						float weight = weightPrev + (error * visibleStatePrev);
						write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weight);
					}
				}
#			endif
		}

		template <int RADIUS>
		static void plLearnPredWeights_v1(
			const Array2D<float> &visibleStatesPrev,
			const Array2D<float> &targets,
			const Array2D<float> &hiddenStatesPrev,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, //write only
			const int2 /*visibleSize*/,
			const float2 hiddenToVisible,
			const float weightAlpha)
		{
			throw 1;
			std::tuple<int2, int2> ranges = cornerCaseRange(hiddenStatesPrev._size, visibleStatesPrev._size, RADIUS, hiddenToVisible);
			const int x0 = 0;
			const int x1 = std::get<0>(ranges).x;
			const int x2 = std::get<0>(ranges).y;
			const int x3 = hiddenStatesPrev._size.x;
			const int y0 = 0;
			const int y1 = std::get<1>(ranges).x;
			const int y2 = std::get<1>(ranges).y;
			const int y3 = hiddenStatesPrev._size.y;

			for (int hiddenPosition_x = x0; hiddenPosition_x < x1; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
			}
		}

		template <int RADIUS>
		static void plLearnPredWeights_v0(
			const Array2D<float> &visibleStatesPrev,
			const Array2D<float> &targets,
			const Array2D<float> &hiddenStatesPrev,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, //write only
			const int2 /*visibleSize*/,
			const float2 hiddenToVisible,
			const float weightAlpha)
		{
			throw 1;
			for (int hiddenPosition_x = 0; hiddenPosition_x < hiddenStatesPrev._size.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < hiddenStatesPrev._size.y; ++hiddenPosition_y) {
					plLearnPredWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenToVisible, weightAlpha, visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront);
				}
			}
		}

		static void plLearnPredWeights(
			const Array2D<float> &visibleStatesPrev,	// in
			const Array2D<float> &targets,				// in
			const Array2D<float> &hiddenStatesPrev,		// in
			const Array3D<float> &weightsBack,			// in
			Array3D<float> &weightsFront,				//write only
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int radius,
			const float alpha,
			const int2 range)
		{
			// last checked: 28-nov 2016
			/*
			switch (radius) {
			case 6: plLearnPredWeights_v1<6>(visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront, visibleSize, hiddenToVisible, weightAlpha); break;
			case 8: plLearnPredWeights_v1<8>(visibleStatesPrev, targets, hiddenStatesPrev, weightsBack, weightsFront, visibleSize, hiddenToVisible, weightAlpha); break;
			default: printf("ERROR: PredictorLayer::plLearnPredWeights: provided radius %i is not implemented\n", radius); break;
			}
			*/
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {

					const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
					const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);

					const float error = read_2D(targets, hiddenPosition_x, hiddenPosition_y) - read_2D(hiddenStatesPrev, hiddenPosition_x, hiddenPosition_y);
					const int fieldLowerBound_x = visiblePositionCenter_x - radius;
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					for (int dx = -radius; dx <= radius; ++dx) {
						const int visiblePosition_x = visiblePositionCenter_x + dx;
						if (inBounds(visiblePosition_x, visibleSize.x)) {
							const int offset_x = visiblePosition_x - fieldLowerBound_x;

							for (int dy = -radius; dy <= radius; ++dy) {
								const int visiblePosition_y = visiblePositionCenter_y + dy;
								if (inBounds(visiblePosition_y, visibleSize.y)) {
									const int offset_y = visiblePosition_y - fieldLowerBound_y;

									const int wi = offset_y + offset_x * (radius * 2 + 1);
									const float weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
									const float visibleStatePrev = read_2D(visibleStatesPrev, visiblePosition_x, visiblePosition_y);
									const float weight = weightPrev + alpha * error * visibleStatePrev;
									write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weight);
								}
							}
						}
					}
				}
			}
		}
	};
}
