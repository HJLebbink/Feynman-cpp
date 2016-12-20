// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <random>
#include <cassert>

#include "Architect.ipp"
#include "Predictor.ipp"
#include "LayerDescs.ipp"
#include "PlotDebug.ipp"

namespace feynman {

	//Default Hierarchy implementation (FeatureHierarchy)
	class Hierarchy {
	private:

		// the following member fields are created in Friend class Architect, method generateHierarchy
		Predictor _p;
		std::mt19937 _rng;
		//std::vector<Array2D<float>> _inputImages;
		std::vector<Array2D<float>> _predictions;
		std::vector<PredictorLayer> _readoutLayers;

	public:

		/*!
		\brief Run a single simulation tick
		\param inputs the inputs to the bottom-most layer.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const std::vector<Array2D<float>> &inputs,
			const bool learn = true)
		{
			// last checked: 28-nov 2016

			if (EXPLAIN) std::cout << "EXPLAIN: Hierarchy:simStep: running Predictor.simStep on " << inputs.size() << " inputs." << std::endl;
			_p.simStep(inputs, inputs, _rng, learn);

			// Get prediction
			if (EXPLAIN) std::cout << "EXPLAIN: Hierarchy:simStep: going to calculate predictions." << std::endl;


			const size_t nPredictions = _predictions.size();
			for (size_t i = 0; i < nPredictions; ++i) {
				//plots::plotImage(_p.getHiddenPrediction()[_back], DEBUG_IMAGE_WIDTH, "Hierarchy:simStep:hiddenPrediction" + std::to_string(i));

				if (EXPLAIN) std::cout << "EXPLAIN: Hierarchy:simStep: prediction layer " << i << "/" << nPredictions << ": running PredictionLayer.activate on the prediction of the hidden layer." << std::endl;
				_readoutLayers[i].activate({ _p.getHiddenPrediction()[_back] }, _rng);

				if (learn) {
					if (EXPLAIN) std::cout << "EXPLAIN: Hierarchy:simStep: prediction layer " << i << "/" << nPredictions << ": running PredictionLayer.learn on input[" << i << "] layer." << std::endl;
					_readoutLayers[i].learn(inputs[i]);
				}
				_readoutLayers[i].stepEnd();
				copy(_readoutLayers[i].getHiddenStates()[_back], _predictions[i]);
				//plots::plotImage(_predictions[i], DEBUG_IMAGE_WIDTH, "Hierarchy:_predictions[" + std::to_string(i)+"]");
			}
		}

		//Get the current prediction vector (that has been created by the last simStep)
		const std::vector<Array2D<float>> &getPredictions() const {
			return _predictions;
		}

		Array2D<float> getFeature(const Array2D<float> &hiddenState) {
			_readoutLayers[0].activate({ hiddenState }, _rng);
			return _readoutLayers[0].getHiddenStates()[_front];
		}

		//Access underlying Predictor
		Predictor &getPredictor() {
			return _p;
		}

		friend class Architect;
	};
}