// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <cassert>

#include "FeatureHierarchy.ipp"
#include "AgentLayer.ipp"
#include "Helpers.ipp"

namespace feynman {

	//Swarm of agents routed through a feature hierarchy
	class AgentSwarm {
	public:

		//Layer desc for swarm layers
		struct AgentLayerDesc {

			//Radius of connections onto previous swarm layer
			int _radius;

			//Q learning parameters
			float _qAlpha;
			float _qGamma;
			float _qLambda;
			float _epsilon;

			//Initialize defaults
			AgentLayerDesc()
				: _radius(12), _qAlpha(0.00004f),
				_qGamma(0.99f), _qLambda(0.98f), _epsilon(0.06f)
			{}
		};

	private:
		//Feature hierarchy with same dimensions as swarm layer
		FeatureHierarchy _h;

		//Layers and descs
		std::vector<AgentLayer> _aLayers;
		std::vector<AgentLayerDesc> _aLayerDescs;

		// All ones image for first layer modulation
		Image2D<float> _ones;

	public:

		//Initialize defaults
		AgentSwarm()
		{}

		/*!
		\brief Create a predictive hierarchy with random initialization.
		Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
		\param cs is the ComputeSystem.
		\param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
		\param inputSize is the (2D) size of the input layer.
		\param actionSize is the (2D) size of the action layer.
		\param actionTileSize is the (2D) size of each action tile (square one-hot action region).
		\param actionRadius is the radius onto the input action layer.
		\param aLayerDescs are Agent layer descriptors.
		\param hLayerDescs are Feature hierarchy layer descriptors.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const int2 inputSize, 
			const int2 actionSize,
			const int2 actionTileSize,
			const int actionRadius,
			const std::vector<AgentLayerDesc> &aLayerDescs,
			const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			assert(aLayerDescs.size() > 0);
			assert(aLayerDescs.size() == hLayerDescs.size());

			// Create underlying hierarchy
			_h.createRandom(std::vector<FeatureHierarchy::InputDesc>{ FeatureHierarchy::InputDesc(inputSize, hLayerDescs.front()._inputDescs.front()._radius), FeatureHierarchy::InputDesc(actionSize, actionRadius) }, hLayerDescs, initWeightRange, rng);

			_aLayerDescs = aLayerDescs;
			_aLayers.resize(_aLayerDescs.size());

			for (int l = 0; l < _aLayers.size(); l++) {
				std::vector<AgentLayer::VisibleLayerDesc> agentVisibleLayerDescs(1);

				agentVisibleLayerDescs[0]._radius = aLayerDescs[l]._radius;
				agentVisibleLayerDescs[0]._alpha = aLayerDescs[l]._qAlpha;
				agentVisibleLayerDescs[0]._size = (l == 0) ? hLayerDescs[l]._size : int2{ hLayerDescs[l]._size.x * 2, hLayerDescs[l]._size.y * 2 };

				_aLayers[l].createRandom((l == _aLayerDescs.size() - 1) ? actionSize : hLayerDescs[l + 1]._size, (l == _aLayerDescs.size() - 1) ? actionTileSize : int2{ 2, 2 }, agentVisibleLayerDescs, initWeightRange, rng);
			}

			_ones = Image2D<float>(actionSize);
			clear(_ones);
		}

		/*!
		\brief Simulation step of hierarchy
		Takes reward and inputs, optionally disable learning.
		\param cs is the ComputeSystem.
		\param reward the reinforcement learning signal.
		\param input the input layer state.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const float reward,
			const Image2D<float> &input,
			std::mt19937 &rng,
			const bool learn) 
		{
			_h.simStep({ input, _aLayers.back().getOneHotActions() }, rng, learn);

			// Update agent layers
			for (int l = 0; l < _aLayers.size(); l++) {
				Image2D<float> feedBack = (l == 0) ? _h.getLayer(l)._sparseFeatures.getHiddenStates()[_back] : _aLayers[l - 1].getOneHotActions();

				if (l == _aLayers.size() - 1)
					_aLayers[l].simStep(reward, std::vector<Image2D<float>>(1, feedBack), _ones, _aLayerDescs[l]._qGamma, _aLayerDescs[l]._qLambda, _aLayerDescs[l]._epsilon, rng, learn);
				else
					_aLayers[l].simStep(reward, std::vector<Image2D<float>>(1, feedBack), _h.getLayer(l + 1)._sparseFeatures.getHiddenStates()[_back], _aLayerDescs[l]._qGamma, _aLayerDescs[l]._qLambda, _aLayerDescs[l]._epsilon, rng, learn);
			}

		}

		/*!
		\brief Get number of agent (swarm) layers
		Matches the number of feature hierarchy layers
		*/
		size_t getNumAgentLayers() const {
			return _aLayers.size();
		}

		/*!
		\brief Get access to an agent layer
		*/
		const AgentLayer &getAgentLayer(int index) const {
			return _aLayers[index];
		}

		/*!
		\brief Get access to an agent layer descriptor
		*/
		const AgentLayerDesc &getAgentLayerDesc(int index) const {
			return _aLayerDescs[index];
		}

		/*!
		\brief Get the actions
		Returns float 2D image where each element is actually an integer, representing the index of the select action for each tile.
		To get continuous values, divide each tile index by the number of elements in a tile (actionTileSize.x * actionTileSize.y).
		*/
		const Image2D<float> &getAction() const {
			return _aLayers.back().getActions()[_back];
		}

		/*!
		\brief Get the underlying feature hierarchy
		*/
		FeatureHierarchy &getHierarchy() {
			return _h;
		}
	};
}