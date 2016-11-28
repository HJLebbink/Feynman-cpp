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
#include "LayerDescs.ipp"
#include "AgentSwarm.ipp"

namespace feynman {

	//Default Agent implementation (AgentSwarm)
	class Agent {
	private:

		//Internal OgmaNeo agent
		AgentSwarm _as;

		std::mt19937 _rng;
		Image2D _inputImage;

		std::vector<Array2D2f> _inputImages;
		std::vector<Image2D> _actions;

	public:

		//Run a single simulation tick
		void simStep(
			const float reward, 
			const std::vector<Array2D2f> &inputs,
			const bool learn = true)
		{
			for (size_t i = 0; i < _inputImages.size(); ++i) {
				copy(inputs[i], _inputImages[i]);
			}
			_as.simStep(reward, _inputImages, _inputImages, _rng, learn);

			// Get actions
			for (size_t i = 0; i < _actions.size(); ++i) {
				copy(_as.getAction(i), _actions[i]);
			}
		}

		//Get the action vector
		const std::vector<Image2D> &getAction() const {
			return _actions;
		}

		friend class Architect;
	};
}