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

		Predictor _p;
		std::mt19937 _rng;
		std::vector<Image2D> _inputImages;
		std::vector<Image2D> _predictions;
		std::vector<PredictorLayer> _readoutLayers;

	public:

		/*!
		\brief Run a single simulation tick
		\param inputs the inputs to the bottom-most layer.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const std::vector<Image2D> &inputs,
			const bool learn = true) 
		{
			// last checked: 24-nov 2016

			// Write input
			for (size_t i = 0; i < _inputImages.size(); ++i) {
				//plots::plotImage(inputs[i], 4, "Hierarchy:simStep:input" + std::to_string(i));
				copy(inputs[i], _inputImages[i]);
			}

			_p.simStep(_inputImages, _inputImages, _rng, learn);

			// Get prediction
			for (size_t i = 0; i < _predictions.size(); ++i) {
				_readoutLayers[i].activate({ _p.getHiddenPrediction()[_back] }, _rng);

				if (learn) {
					_readoutLayers[i].learn(_inputImages[i]);
				}
				_readoutLayers[i].stepEnd();
				copy(_readoutLayers[i].getHiddenStates()[_back], _predictions[i]);
				//plots::plotImage(_predictions[i], 4, "Hierarchy:pred" + std::to_string(i));
			}
		}

		//Get the current prediction vector
		const std::vector<Image2D> &getPredictions() const {
			return _predictions;
		}

		//Access underlying Predictor
		Predictor &getPredictor() {
			return _p;
		}

		friend class Architect;
	};
}