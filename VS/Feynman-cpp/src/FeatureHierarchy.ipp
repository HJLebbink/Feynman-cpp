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
#include <algorithm> // for std::max

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
			DoubleBuffer2D<float> _tpBuffer;

			//brief Prediction error temporary buffer
			Array2D<float> _predErrors;

			//Flag (for use by other systems) to reset temporal pooler
			bool _tpReset;
			//Flag (for use by other systems) to reset temporal pooler
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
				_layers[layer]._tpBuffer = createDoubleBuffer2D<float>(_layers[layer]._sf->getHiddenSize());

				// Prediction error
				_layers[layer]._predErrors = Array2D<float>(_layers[layer]._sf->getHiddenSize());
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
			const std::vector<Array2D<float>> &inputs,
			const std::vector<Array2D<float>> &predictionsPrev,
			std::mt19937 &rng,
			const bool learn = true)
		{
			// last checked: 28-nov-2016

			// Clear summation buffers if reset previously
			for (size_t l = 0; l < _layers.size(); ++l) {
				if (_layers[l]._tpNextReset) {
					if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": clearing temporal pooling buffer (" << _layerDescs[l]._poolSteps << ")." << std::endl;
					clear(_layers[l]._tpBuffer[_back]);
				}
				else {
					if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": not clearing temporal pooling buffer (" << _layerDescs[l]._poolSteps << ")." << std::endl;
				}
			}

			// Activate
			bool prevClockReset = true;

			for (size_t l = 0; l < _layers.size(); ++l) {
				// Add input to pool
				if (prevClockReset) {
					_layers[l]._clock++;

					// Gather inputs for layer
					std::vector<Array2D<float>> visibleStates;
					{
						if (l == 0) {
							std::vector<Array2D<float>> inputsUse = inputs;

							if (_layerDescs[0]._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent) {
								visibleStates = inputs;
								visibleStates.push_back(_layers[0]._sf->getHiddenContext());
								if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": running sparseFeature.activate on 1) " << inputs.size() << " input layers and 2) recurrent influx from hidden state from layer " << l << "." << std::endl;
							}
							else {
								visibleStates = inputs;
								if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": running sparseFeature.activate on 1) " << inputs.size() << " input layers (and no recurrent influx)." << std::endl;
							}
						}
						else {
							if (_layerDescs[l]._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent) {
								visibleStates = { _layers[l - 1]._tpBuffer[_back], _layers[l]._sf->getHiddenContext() };
								if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": running sparseFeature.activate on 1) temporal pooled hidden layer " << (l - 1) << " and 2) recurrent influx from hidden state from layer " << l << "." << std::endl;
							}
							else {
								visibleStates = { _layers[l - 1]._tpBuffer[_back] };
								if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": running sparseFeature.activate on 1) temporal pooled hidden layer " << (l - 1) << " (and no recurrent influx)." << std::endl;
							}
						}
					}

					_layers[l]._sf->activate(visibleStates, predictionsPrev[l], rng);

					if (learn) {
						if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": running sparseFeature.learn." << std::endl;
						_layers[l]._sf->learn(rng);
					}
					_layers[l]._sf->stepEnd();

					// Prediction error
					//plots::plotImage(_layers[l]._sf->getHiddenStates()[_back], 8, "FeatureHierarchy:simStep:hiddenState" + std::to_string(l));

					if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": calculating prediction error between hidden state and previous prediction." << std::endl;
					fhPredError(
						_layers[l]._sf->getHiddenContext(),	// in
						predictionsPrev[l],					// in
						_layers[l]._predErrors,				// out
						_layers[l]._sf->getHiddenSize()
					);

					//plots::plotImage(_layers[l]._predErrors, 8, "FeatureHierarchy:simStep:predErrors" + std::to_string(l));

					// Add state to average
					if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": temporal pooling ("<< _layerDescs[l]._poolSteps << ") prediction errors." << std::endl;
					fhPool(
						_layers[l]._predErrors,				// in
						_layers[l]._tpBuffer[_back],		// in
						_layers[l]._tpBuffer[_front],		// out
						1.0f / std::max(1, _layerDescs[l]._poolSteps),
						_layers[l]._sf->getHiddenSize()
					);

					std::swap(_layers[l]._tpBuffer[_front], _layers[l]._tpBuffer[_back]);
				}
				else {
					if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:simStep: layer " << l << "/" << _layers.size() << ": previous clock is not reset." << std::endl;
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
			const Array2D<float> &states,		// in
			const Array2D<float> &outputsBack,	// in
			Array2D<float> &outputsFront,		// out
			const float scale,					// not used
			const int2 range)
		{
			if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:fhPool: scale=" << scale << std::endl;
			// max pooling

			// last checked: 06-dec-2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float state = states._data_float[i];
				const float outputPrev = outputsBack._data_float[i];
				const float newValue = (outputPrev > state) ? outputPrev : state;
				outputsFront._data_float[i] = newValue;
			}
		}

		static void fhPredError(
			const Array2D<float> &states,			// in
			const Array2D<float> &predictionsPrev,	// in
			Array2D<float> &errors,					// out
			const int2 range)
		{
			if (EXPLAIN) std::cout << "EXPLAIN: FeatureHierarchy:fhPredError." << std::endl;

			// last checked: 28-nov-2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float state = states._data_float[i];
				const float predictionPrev = predictionsPrev._data_float[i];
				//write_imagef(errors, position, (float4)(state - predictionPrev, 0.0f, 0.0f, 0.0f));
				//write_imagef(errors, position, (float4)(state, 0.0f, 0.0f, 0.0f));
				const float newValue = (state * (1.0f - predictionPrev)) + (predictionPrev * (1.0f - state));
				errors._data_float[i] = newValue;
			}
		}
	};
}