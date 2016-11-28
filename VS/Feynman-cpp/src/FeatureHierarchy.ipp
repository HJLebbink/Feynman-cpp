// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <random>

#include "Helpers.ipp"
#include "SparseFeatures.ipp"
#include "PlotDebug.ipp"

namespace feynman {

	//Hierarchy of sparse features
	class FeatureHierarchy {
	public:

		//Descriptor of a layer in the feature hierarchy
		struct LayerDesc {
			//Sparse features desc
			std::shared_ptr<SparseFeatures::SparseFeaturesDesc> _sfDesc;

			// Temporal pooling
			int _poolSteps;

			//Initialize defaults
			LayerDesc()
				: _poolSteps(2)
			{}
		};

		//brief Layer
		struct Layer {
			//Sparse features
			std::shared_ptr<SparseFeatures> _sf;

			//Clock for temporal pooling (relative to previous layer)
			int _clock;

			//brief Temporal pooling buffer
			DoubleBuffer2D2f _tpBuffer;

			//brief Prediction error temporary buffer
			Image2D _predErrors;

			//Flags for use by other systems
			bool _tpReset;
			bool _tpNextReset;

			//Initialize defaults
			Layer()
				: _clock(0), _tpReset(false), _tpNextReset(false)
			{}
		};

	private:

		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

	public:
		//Initialize defaults
		FeatureHierarchy()
		{}

		/**
		Create a sparse feature hierarchy with random initialization.
		Requires initialization information.
		\param inputDescs are the descriptors of the input layers.
		\param layerDescs are descriptors for feature hierarchy layers.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const std::vector<LayerDesc> &layerDescs,
			std::mt19937 &rng)
		{
			// last checked: 25-nov-2016
			_layerDescs = layerDescs;
			_layers.resize(_layerDescs.size());

			for (size_t layer = 0; layer < _layers.size(); ++layer) {
				//std::cout << "INFO: layer=" << layer << std::endl;
				_layers[layer]._sf = _layerDescs[layer]._sfDesc->sparseFeaturesFactory();

				// Create temporal pooling buffer
				_layers[layer]._tpBuffer = createDoubleBuffer2D2f(_layers[layer]._sf->getHiddenSize());

				// Prediction error
				_layers[layer]._predErrors = Image2D(_layers[layer]._sf->getHiddenSize());
			}
		}

		/*!
		\brief Simulation step of hierarchy
		Runs one timestep of simulation.
		\param inputs the inputs to the bottom-most layer.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const std::vector<Array2D2f> &inputs,
			const std::vector<Array2D2f> &predictionsPrev,
			std::mt19937 &rng,
			const bool learn = true)
		{
			// last checked: 28-nov-2016

			// Clear summation buffers if reset previously
			for (size_t l = 0; l < _layers.size(); ++l) {
				if (_layers[l]._tpNextReset) {
					// Clear summation buffer
					clear(_layers[l]._tpBuffer[_back]);
				}
			}

			// Activate
			bool prevClockReset = true;

			for (size_t l = 0; l < _layers.size(); ++l) {
				// Add input to pool
				if (prevClockReset) {
					_layers[l]._clock++;

					// Gather inputs for layer
					std::vector<Array2D2f> visibleStates;
					{
						if (l == 0) {
							std::vector<Array2D2f> inputsUse = inputs;

							if (_layerDescs.front()._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent)
								inputsUse.push_back(_layers.front()._sf->getHiddenContext());

							visibleStates = inputsUse;
						}
						else
							visibleStates = (_layerDescs[l]._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent) 
								? std::vector<Array2D2f>{ _layers[l - 1]._tpBuffer[_back], _layers[l]._sf->getHiddenContext() }
								: std::vector<Array2D2f>{ _layers[l - 1]._tpBuffer[_back] };
					}

					// Update layer
					_layers[l]._sf->activate(visibleStates, predictionsPrev[l], rng);

					if (learn)
						_layers[l]._sf->learn(rng);

					_layers[l]._sf->stepEnd();

					// Prediction error
					//plots::plotImage(_layers[l]._sf->getHiddenStates()[_back], 8, "FeatureHierarchy:simStep:hiddenState" + std::to_string(l));

					fhPredError(
						_layers[l]._sf->getHiddenStates()[_back],	// in
						predictionsPrev[l],							// in
						_layers[l]._predErrors,						// out
						_layers[l]._sf->getHiddenSize()
					);

					//plots::plotImage(_layers[l]._predErrors, 8, "FeatureHierarchy:simStep:predErrors" + std::to_string(l));

					// Add state to average
					fhPool(
						_layers[l]._predErrors,
						_layers[l]._tpBuffer[_back],
						_layers[l]._tpBuffer[_front],
						1.0f / std::max(1, _layerDescs[l]._poolSteps),
						_layers[l]._sf->getHiddenSize()
					);

					std::swap(_layers[l]._tpBuffer[_front], _layers[l]._tpBuffer[_back]);
				}

				_layers[l]._tpReset = prevClockReset;

				if (_layers[l]._clock >= _layerDescs[l]._poolSteps) {
					_layers[l]._clock = 0;

					prevClockReset = true;
				}
				else
					prevClockReset = false;

				_layers[l]._tpNextReset = prevClockReset;
			}
		}

		// Get number of layers
		size_t getNumLayers() const {
			return _layers.size();
		}

		/*!
		\brief Get access to a layer
		\param[in] index Layer index.
		*/
		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		/*!
		\brief Get access to a layer desc
		\param[in] index Layer index.
		*/
		const LayerDesc &getLayerDesc(int index) const {
			return _layerDescs[index];
		}

		//Clear the working memory
		void clearMemory() {
			// last checked: 25-nov-2016
			for (size_t layer = 0; layer < _layers.size(); ++layer) {
				_layers[layer]._sf->clearMemory();
			}
		}

	private:

		static void fhPool(
			const Image2D &states,
			const Array2D2f &outputsBack,
			Array2D2f &outputsFront,
			const float scale,
			const int2 range)
		{
			// last checked: 28-nov-2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float state = states._data_float[i];
				const float outputPrev = outputsBack._data_float[i].x;
				const float newValue = (outputPrev > state) ? outputPrev : state;
				outputsFront._data_float[i] = { newValue, 0.0f };
			}
		}

		static void fhPredError(
			const Array2D2f &states,
			const Array2D2f &predictionsPrev,
			Image2D &errors,
			const int2 range)
		{
			// last checked: 28-nov-2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float state = states._data_float[i].x;
				const float predictionPrev = predictionsPrev._data_float[i].x;
				//write_imagef(errors, position, (float4)(state - predictionPrev, 0.0f, 0.0f, 0.0f));
				//write_imagef(errors, position, (float4)(state, 0.0f, 0.0f, 0.0f));
				const float newValue = state * (1.0f - predictionPrev) + (1.0f - state) * predictionPrev;
				errors._data_float[i] = newValue;
			}
		}
	};
}