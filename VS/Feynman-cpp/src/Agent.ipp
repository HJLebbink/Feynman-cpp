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
		AgentSwarm _agentSwarm;

		int _inputWidth, _inputHeight;
		int _actionWidth, _actionHeight;
		int _actionTileWidth, _actionTileHeight;

		std::mt19937 _rng;
		Image2D _inputImage;
		std::vector<float> _action; // Exploratory action (normalized float [0, 1])

	public:
		/*!
		\brief Create the Agent
		\param inputWidth is the (2D) width of the input layer.
		\param inputHeight is the (2D) height of the input layer.
		\param actionWidth is the (2D) width of the action layer.
		\param actionHeight is the (2D) height of the action layer.
		\param actionTileWidth is the (2D) width of each action tile (square one-hot action region).
		\param actionTileHeight is the (2D) height of each action tile (square one-hot action region).
		\param actionRadius is the radius onto the input action layer.
		\param layerDescs provide layer descriptors for hierachy and agent.
		\param initMinWeight is the minimum value for weight initialization.
		\param initMaxWeight is the maximum value for weight initialization.
		\param seed a random number generator seed.
		*/
		Agent(
			const int inputWidth,
			const int inputHeight,
			const int actionWidth,
			const int actionHeight,
			const int actionTileWidth,
			const int actionTileHeight,
			const int actionRadius,
			const std::vector<LayerDescs> &layerDescs,
			const float initMinWeight,
			const float initMaxWeight,
			const int seed)
		{
			_rng.seed(seed);

			_inputWidth = inputWidth;
			_inputHeight = inputHeight;
			_actionWidth = actionWidth;
			_actionHeight = actionHeight;
			_actionTileWidth = actionTileWidth;
			_actionTileHeight = actionTileHeight;

			int2 inputSize = { inputWidth, inputHeight };
			int2 actionSize = { actionWidth, actionHeight };

			std::vector<AgentSwarm::AgentLayerDesc> aLayerDescs(layerDescs.size());
			std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(layerDescs.size());

			for (size_t layer = 0; layer < layerDescs.size(); layer++) {
				hLayerDescs[layer]._size = int2{ layerDescs[layer]._width, layerDescs[layer]._height };
				hLayerDescs[layer]._inputDescs = { FeatureHierarchy::InputDesc(inputSize, layerDescs[layer]._feedForwardRadius) };
				hLayerDescs[layer]._inhibitionRadius = layerDescs[layer]._inhibitionRadius;
				hLayerDescs[layer]._recurrentRadius = layerDescs[layer]._recurrentRadius;
				hLayerDescs[layer]._spFeedForwardWeightAlpha = layerDescs[layer]._spFeedForwardWeightAlpha;
				hLayerDescs[layer]._spRecurrentWeightAlpha = layerDescs[layer]._spRecurrentWeightAlpha;
				hLayerDescs[layer]._spBiasAlpha = layerDescs[layer]._spBiasAlpha;
				hLayerDescs[layer]._spActiveRatio = layerDescs[layer]._spActiveRatio;

				aLayerDescs[layer]._radius = layerDescs[layer]._qRadius;
				aLayerDescs[layer]._qAlpha = layerDescs[layer]._qAlpha;
				aLayerDescs[layer]._qGamma = layerDescs[layer]._qGamma;
				aLayerDescs[layer]._qLambda = layerDescs[layer]._qLambda;
				aLayerDescs[layer]._epsilon = layerDescs[layer]._epsilon;
			}

			_agentSwarm.createRandom(inputSize, actionSize, { actionTileWidth, actionTileHeight }, actionRadius, aLayerDescs, hLayerDescs, float2{ initMinWeight, initMaxWeight }, _rng);

			// Create temporary buffers
			_inputImage = Image2D(int2{ inputWidth, inputHeight });

			_action.clear();
			_action.assign(actionWidth * actionHeight, 0.0f);
		}

		//Run a single simulation tick
		void simStep(
			const float reward, 
			const std::vector<float> &inputs,
			const bool learn)
		{
			assert(inputs.size() == _inputWidth * _inputHeight);

			// Write input
			copy(inputs, _inputImage);

			// Do the sim step
			_agentSwarm.simStep(reward, _inputImage, _rng, learn);

			// Get action
			copy(_agentSwarm.getAction(), _action);
		}

		//Get the action vector
		const std::vector<float> &getAction() const {
			return _action;
		}

		/*!
		\brief Get the hidden states for a layer
		\param[in] layerIndex Layer index.
		*/
		std::vector<float> getStates(int layerIndex) {
			std::vector<float> states(_agentSwarm.getHierarchy().getLayerDesc(layerIndex)._size.x * _agentSwarm.getHierarchy().getLayerDesc(layerIndex)._size.y * 2);
			//_pCs->getQueue().enqueueReadImage(_agentSwarm.getHierarchy().getLayer(layerIndex)._sp.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_agentSwarm.getHierarchy().getLayerDesc(layerIndex)._size.x), static_cast<cl::size_type>(_agentSwarm.getHierarchy().getLayerDesc(layerIndex)._size.y), 1 }, 0, 0, states.data());
			Image2D src = _agentSwarm.getHierarchy().getLayer(layerIndex)._sparseFeatures.getHiddenStates()[_back];
			return src._data;
		}
	};
}