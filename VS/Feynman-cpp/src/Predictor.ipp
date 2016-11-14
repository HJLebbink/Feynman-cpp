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
#include <cassert>
#include <string>

#include "Helpers.ipp"
#include "FeatureHierarchy.ipp"
#include "PredictorLayer.ipp"

namespace feynman {

	//Predicts temporal streams of data. Combines the bottom-up feature hierarchy with top-down predictions.
	class Predictor {

	public:
		//Description of a predictor layer
		struct PredLayerDesc {
			//Predictor layer properties. Radius onto hidden layer, learning rates for feed-forward and feed-back.
			int _radius;
			float _alpha;
			float _beta;

			//Initialize defaults
			PredLayerDesc()
				: _radius(8), _alpha(0.08f), _beta(0.16f)
			{}
		};

	private:

		FeatureHierarchy _featureHierarchy;
		int2 _inputSize;
		std::vector<PredLayerDesc> _pLayerDescs;
		std::vector<PredictorLayer> _predictorLayers;

	public:
		/*!
		\brief Create a sparse predictive hierarchy with random initialization.
		Requires initialization information.
		\param inputSize size of the (2D) input.
		\param pLayerDescs Predictor layer descriptors.
		\param hLayerDescs Feature hierarchy layer descriptors.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		\param firstLearningRateScalar since the first layer predicts without thresholding while all others predict with it,
		the learning rate is scaled by this parameter for that first layer. Set to 1 if you want your pre-set learning rate
		to remain unchanged.
		*/
		void createRandom(
			const int2 inputSize, 
			const std::vector<PredLayerDesc> &pLayerDescs, 
			const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
			const float2 initWeightRange,
			std::mt19937 &rng, 
			float firstLearningRateScalar = 0.1f)
		{
			if ((pLayerDescs.size() != hLayerDescs.size()) | pLayerDescs.empty())
			{
				std::cout << "WARNING: Predictor:createRandom: pLayerDescs.size()" << pLayerDescs.size() << "; hLayerDescs.size()=" << hLayerDescs.size() << std::endl;
				throw 1;
			}

			_inputSize = inputSize;
			// Create underlying hierarchy
			_featureHierarchy.createRandom(
				std::vector<FeatureHierarchy::InputDesc>{ FeatureHierarchy::InputDesc(inputSize, hLayerDescs.front()._inputDescs.front()._radius) }, 
				hLayerDescs, 
				initWeightRange, 
				rng
			);

			int2 prevLayerSize = inputSize;
			_pLayerDescs = pLayerDescs;
			_predictorLayers.resize(_pLayerDescs.size());

			for (size_t layer = 0; layer < _predictorLayers.size(); layer++) {
				std::vector<PredictorLayer::VisibleLayerDesc> pVisibleLayerDescs;

				const float alpha = (layer == 0) ? (firstLearningRateScalar * _pLayerDescs[layer]._alpha) : _pLayerDescs[layer]._alpha;
				const float beta  = (layer == 0) ? (firstLearningRateScalar * _pLayerDescs[layer]._beta)  : _pLayerDescs[layer]._beta;

				if (layer < _predictorLayers.size() - 1) {
					pVisibleLayerDescs.resize(2);

					// Current
					pVisibleLayerDescs[0]._radius = _pLayerDescs[layer]._radius;
					pVisibleLayerDescs[0]._alpha = alpha;
					pVisibleLayerDescs[0]._size = hLayerDescs[layer]._size;

					// Feed back
					pVisibleLayerDescs[1]._radius = _pLayerDescs[layer]._radius;
					pVisibleLayerDescs[1]._alpha = beta;
					pVisibleLayerDescs[1]._size = hLayerDescs[layer]._size;
				}
				else {
					pVisibleLayerDescs.resize(1);

					// Current
					pVisibleLayerDescs[0]._radius = _pLayerDescs[layer]._radius;
					pVisibleLayerDescs[0]._alpha = alpha;
					pVisibleLayerDescs[0]._size = hLayerDescs[layer]._size;
				}

				_predictorLayers[layer].createRandom(prevLayerSize, pVisibleLayerDescs, initWeightRange, rng);

				// Next layer
				prevLayerSize = hLayerDescs[layer]._size;
			}
		}
		
		/*!
		\brief Simulation step of hierarchy
		\param input input to the hierarchy (2D).
		\param inputCorrupted in many cases you can pass in the same value as for input, but in some cases you can also pass
		a corrupted version of the input for a "denoising auto-encoder" style effect on the learned weights.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const Image2D &input,
			const Image2D &inputCorrupted,
			std::mt19937 &rng,
			const bool learn = true)
		{
			//plots::plotImage(inputCorrupted, 8.0f, false, "Predictor:inputCorrupted");
			//TODO: consider using binary input instead of float
			//const float2 minmax = find_min_max(inputCorrupted);
			//printf("INFO: Predictor:simStep, min=%f; max=%f\n", minmax.x, minmax.y);

			// Activate hierarchy
			_featureHierarchy.simStep({ inputCorrupted }, rng, learn);
			const int nLayers = static_cast<int>(_predictorLayers.size());

			// Forward pass through predictor to get next prediction
			for (int layer = (nLayers - 1); (layer >= 0); --layer) {

				if (true) { // display hidden layer
					const Image2D &layerData = _featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_back];
					const float desiredSize = 400.0f;
					const float rescaleSize = desiredSize / layerData._size.x;
					plots::plotImage(layerData, rescaleSize, "Hidden.Layer" + std::to_string(layer) + " ("+std::to_string(layerData._size.x) + "x" + std::to_string(layerData._size.y) +")");
					//const float2 minmax = find_min_max(_featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_back]);
					//printf("INFO: Predictor:simStep, min=%f; max=%f\n", minmax.x, minmax.y);
				}

				const std::vector<Image2D> visibleStates =
					(layer == (nLayers - 1))
						? std::vector<Image2D> { // top-layer only get feature input from the top layer
							_featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_back]
						}
						: std::vector<Image2D>{ // non-top-layers get feature inputs from the current layer and the layer above.
							_featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_back],
							_predictorLayers[layer + 1].getHiddenStates()[_front]
						};

						_predictorLayers[layer].activate(visibleStates, layer != 0);
			}

			if (learn) {
				for (int layer = (nLayers - 1); (layer >= 0); --layer) {

					const Image2D target = 
						(layer == 0) 
							? input 
							: _featureHierarchy.getLayer(layer - 1)._sparseFeatures.getHiddenStates()[_back];

					const std::vector<Image2D> visibleStatesPrev =
						(layer == (nLayers - 1))
							? std::vector<Image2D> { 
								_featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_front] 
							}
							: std::vector<Image2D> {
								_featureHierarchy.getLayer(layer)._sparseFeatures.getHiddenStates()[_front],
								_predictorLayers[layer + 1].getHiddenStates()[_back]
							};

					_predictorLayers[layer].learn(target, visibleStatesPrev);
				}
			}

			for (int layer = 0; layer < nLayers; layer++) {
				_predictorLayers[layer].stepEnd();
			}
		}

		/*!
		\brief Get number of predictor layers
		Matches the number of layers in the feature hierarchy.
		*/
		size_t getNumPLayers() const {
			return _predictorLayers.size();
		}

		//Get access to a predictor layer
		const PredictorLayer &getPLayer(int index) const {
			return _predictorLayers[index];
		}

		//Get access to a predictor layer desc
		const PredLayerDesc &getPLayerDesc(int index) const {
			return _pLayerDescs[index];
		}

		// Get the predictions
		const Image2D &getPrediction() const {
			return _predictorLayers.front().getHiddenStates()[_back];
		}

		//Get the underlying feature hierarchy
		FeatureHierarchy &getHierarchy() {
			return _featureHierarchy;
		}

		// approx memory usage in bytes;
		size_t getMemoryUsage(bool plot) const {
			size_t nBytes = 0;
			for (int layer = 0; layer < static_cast<int>(_featureHierarchy.getNumLayers()); ++layer) {
				if (plot) std::cout << "Predictor: layer" << layer << std::endl;
				nBytes += _featureHierarchy.getLayer(layer)._sparseFeatures.getMemoryUsage(plot);
			}
			if (plot) std::cout << "memory usage " << nBytes << " bytes = " << nBytes / 1024 << " kb = " << nBytes / (1024 * 1024) << " mb" << std::endl;
			return nBytes;
		}
	};
}