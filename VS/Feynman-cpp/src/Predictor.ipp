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
#include "PlotDebug.ipp"


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

		FeatureHierarchy _h;
		std::vector<PredLayerDesc> _pLayerDescs;
		std::vector<PredictorLayer> _pLayers;

	public:
		/*!
		\brief Create a sparse predictive hierarchy with random initialization.
		\brief shouldPredictInput describes which of the bottom (input) layers should be predicted (have an associated predictor layer).
		\param pLayerDescs Predictor layer descriptors.
		\param hLayerDescs Feature hierarchy layer descriptors.
		\param rng a random number generator.
		*/
		void createRandom(
			const std::vector<PredLayerDesc> &pLayerDescs, 
			const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			// last checked: 24-nov 2016
			if ((pLayerDescs.size() != hLayerDescs.size()) | pLayerDescs.empty())
			{
				std::cout << "WARNING: Predictor:createRandom: pLayerDescs.size()" << pLayerDescs.size() << "; hLayerDescs.size()=" << hLayerDescs.size() << std::endl;
				throw 1;
			}

			// Create underlying hierarchy
			_h.createRandom(hLayerDescs, rng);
			_pLayerDescs = pLayerDescs;
			_pLayers.resize(_pLayerDescs.size());

			for (size_t l = 0; l < _pLayers.size(); ++l) {
				std::vector<PredictorLayer::VisibleLayerDesc> pVisibleLayerDescs((l == _pLayers.size() - 1) ? 1 : 2);

				for (size_t p = 0; p < pVisibleLayerDescs.size(); ++p) {
					// Current
					pVisibleLayerDescs[p]._radius = _pLayerDescs[l]._radius;
					pVisibleLayerDescs[p]._alpha = (p == 0) ? _pLayerDescs[l]._alpha : _pLayerDescs[l]._beta;
					pVisibleLayerDescs[p]._size = _h.getLayer(l)._sf->getHiddenSize();
				}

				_pLayers[l].createRandom(
					_h.getLayer(l)._sf->getHiddenSize(), 
					pVisibleLayerDescs, 
					_h.getLayer(l)._sf, 
					initWeightRange, 
					rng
				);
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
			const std::vector<Array2D<float>> /*&inputs*/, // ununused
			const std::vector<Array2D<float>> &inputsCorrupted,
			std::mt19937 &rng,
			const bool learn = true)
		{
			// last checked: 28-nov 2016
			//TODO: consider using binary input instead of float

			std::vector<Array2D<float>> predictionsPrev(_pLayers.size());

			for (size_t l = 0; l < predictionsPrev.size(); ++l) {
				predictionsPrev[l] = _pLayers[l].getHiddenStates()[_back];
				//plots::plotImage(predictionsPrev[l], 6, "Predictor:simStep:predictionsPrev" + std::to_string(l));
			}

			//plots::plotImage(inputsCorrupted[0], 6, "Predictor:simStep:inputsCorrupted");

			// Activate hierarchy
			_h.simStep(inputsCorrupted, predictionsPrev, rng, learn);
			const int nLayers = static_cast<int>(_pLayers.size());

			// Forward pass through predictor to get next prediction
			for (int layer = (nLayers - 1); (layer >= 0); layer--) 
			{
				//plots::plotImage(_h.getLayer(layer)._sf->getHiddenStates()[_back], 6, "Predictor:simStep: Hidden.Layer" + std::to_string(layer));

				//std::cout << "INFO: Predictor:simStep: pass forward layer " << l << std::endl;
				if (_h.getLayer(layer)._tpReset || _h.getLayer(layer)._tpNextReset) 
				{
					const Array2D<float> target = _h.getLayer(layer)._sf->getHiddenStates()[_back];
					//plots::plotImage(target, 6, "Predictor:simStep:target" + std::to_string(l));

					const std::vector<Array2D<float>> inputsNextLayer =
						(layer == (nLayers - 1))
							? std::vector<Array2D<float>>{ target } // top-layer only get feature input from the top layer
							: std::vector<Array2D<float>>{ target, _pLayers[layer + 1].getHiddenStates()[_back] }; // non-top-layers get feature inputs from the current layer and the layer above.
					//std::cout << "INFO: Predictor:simStep: pass forward layer " << l << ";inputsNextLayer.size="<< inputsNextLayer.size() << std::endl;

					_pLayers[layer].activate(inputsNextLayer, rng);
					if (learn) _pLayers[layer].learn(target);
					_pLayers[layer].stepEnd();
				}
			}
		}

		/*!
		\brief Get number of predictor layers
		Matches the number of layers in the feature hierarchy.
		*/
		size_t getNumPLayers() const {
			return _pLayers.size();
		}

		//Get access to a predictor layer
		const PredictorLayer &getPredLayer(int index) const {
			return _pLayers[index];
		}

		//Get access to a predictor layer desc
		const PredLayerDesc &getPredLayerDesc(int index) const {
			return _pLayerDescs[index];
		}

		//Get the predictions
		const DoubleBuffer2D<float> &getHiddenPrediction() const {
			// last checked: 28-nov 2016
			return _pLayers.front().getHiddenStates();
		}


		//Get the underlying feature hierarchy
		FeatureHierarchy &getHierarchy() {
			return _h;
		}

		// approx memory usage in bytes;
		size_t getMemoryUsage(bool plot) const {
			size_t nBytes = 0;
			for (int layer = 0; layer < static_cast<int>(_h.getNumLayers()); ++layer) {
				if (plot) std::cout << "Predictor: layer" << layer << std::endl;
				//nBytes += _featureHierarchy.getLayer(layer)._sf.getMemoryUsage(plot);
			}
			if (plot) std::cout << "memory usage " << nBytes << " bytes = " << nBytes / 1024 << " kb = " << nBytes / (1024 * 1024) << " mb" << std::endl;
			return nBytes;
		}
	};
}