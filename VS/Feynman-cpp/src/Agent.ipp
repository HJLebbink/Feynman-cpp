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

		int _inputWidth, _inputHeight;
		int _actionWidth, _actionHeight;
		int _actionTileWidth, _actionTileHeight;

		std::mt19937 _rng;
		Image2D<float> _inputImage;
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

			for (int l = 0; l < layerDescs.size(); l++) {
				hLayerDescs[l]._size = int2{ layerDescs[l]._width, layerDescs[l]._height };
				hLayerDescs[l]._inputDescs = { FeatureHierarchy::InputDesc(inputSize, layerDescs[l]._feedForwardRadius) };
				hLayerDescs[l]._inhibitionRadius = layerDescs[l]._inhibitionRadius;
				hLayerDescs[l]._recurrentRadius = layerDescs[l]._recurrentRadius;
				hLayerDescs[l]._spFeedForwardWeightAlpha = layerDescs[l]._spFeedForwardWeightAlpha;
				hLayerDescs[l]._spRecurrentWeightAlpha = layerDescs[l]._spRecurrentWeightAlpha;
				hLayerDescs[l]._spBiasAlpha = layerDescs[l]._spBiasAlpha;
				hLayerDescs[l]._spActiveRatio = layerDescs[l]._spActiveRatio;

				aLayerDescs[l]._radius = layerDescs[l]._qRadius;
				aLayerDescs[l]._qAlpha = layerDescs[l]._qAlpha;
				aLayerDescs[l]._qGamma = layerDescs[l]._qGamma;
				aLayerDescs[l]._qLambda = layerDescs[l]._qLambda;
				aLayerDescs[l]._epsilon = layerDescs[l]._epsilon;
			}

			_as.createRandom(inputSize, actionSize, { actionTileWidth, actionTileHeight }, actionRadius, aLayerDescs, hLayerDescs, float2{ initMinWeight, initMaxWeight }, _rng);

			// Create temporary buffers
			_inputImage = Image2D<float>(int2{ inputWidth, inputHeight });

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
			//std::vector<float> inputsf = inputs;
			//_pCs->getQueue().enqueueWriteImage(_inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 }, 0, 0, inputsf.data());
			copy(inputs, _inputImage);

			_as.simStep(reward, _inputImage, _rng, learn);

			// Get action
			//_pCs->getQueue().enqueueReadImage(_as.getAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionWidth), static_cast<cl::size_type>(_actionHeight), 1 }, 0, 0, _action.data());
			copy(_as.getAction(), _action);
		}

		//Get the action vector
		const std::vector<float> &getAction() const {
			return _action;
		}

		/*!
		\brief Get the hidden states for a layer
		\param[in] li Layer index.
		*/
		std::vector<float> getStates(int li) {
			std::vector<float> states(_as.getHierarchy().getLayerDesc(li)._size.x * _as.getHierarchy().getLayerDesc(li)._size.y * 2);
			//_pCs->getQueue().enqueueReadImage(_as.getHierarchy().getLayer(li)._sp.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_as.getHierarchy().getLayerDesc(li)._size.x), static_cast<cl::size_type>(_as.getHierarchy().getLayerDesc(li)._size.y), 1 }, 0, 0, states.data());
			Image2D<float> src = _as.getHierarchy().getLayer(li)._sparseFeatures.getHiddenStates()[_back];
			return src._data;
		}
	};
}