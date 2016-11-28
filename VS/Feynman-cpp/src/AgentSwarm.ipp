// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <cassert>

#include "Helpers.ipp"
#include "FeatureHierarchy.ipp"
#include "AgentLayer.ipp"

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
			float _actionAlpha;
			float _qGamma;
			float _qLambda;
			float _actionLambda;
			float _maxActionWeightMag;

			//Initialize defaults
			AgentLayerDesc()
				: _radius(12), _qAlpha(0.01f), _actionAlpha(0.1f),
				_qGamma(0.99f), _qLambda(0.98f), _actionLambda(0.98f),
				_maxActionWeightMag(10.0f)
			{}
		};

	private:

		Predictor _p;

		//Layers and descs
		std::vector<std::vector<AgentLayer>> _aLayers;
		std::vector<std::vector<AgentLayerDesc>> _aLayerDescs;
		std::vector<float> _rewardSums;
		std::vector<float> _rewardCounts;

		// All ones image for first layer modulation
		std::vector<Array2D2f> _ones;

	public:

		//Initialize defaults
		AgentSwarm()
		{}

		/*!
		\brief Create a predictive hierarchy with random initialization.
		Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
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
			const std::vector<int2> &actionSizes, 
			const std::vector<int2> actionTileSizes,
			const std::vector<std::vector<AgentLayerDesc>> &aLayerDescs,
			const std::vector<Predictor::PredLayerDesc> &pLayerDescs,
			const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
			float2 initWeightRange, std::mt19937 &rng)
		{
			assert(aLayerDescs.size() > 0);
			assert(aLayerDescs.size() == hLayerDescs.size());

			// Create underlying hierarchy
			_p.createRandom(pLayerDescs, hLayerDescs, initWeightRange, rng);

			_aLayerDescs = aLayerDescs;
			_aLayers.resize(_aLayerDescs.size());

			for (size_t layer = 0; layer < _aLayers.size(); ++layer) 
			{
				_aLayers[layer].resize(_aLayerDescs[layer].size());
				for (size_t i = 0; i < _aLayers[layer].size(); i++)
				{
					std::vector<AgentLayer::VisibleLayerDesc> agentVisibleLayerDescs(1);
					const float lrScalar = (layer == 0) ? 0.25f : 1.0f;

					agentVisibleLayerDescs[0]._radius = aLayerDescs[layer][i]._radius;
					agentVisibleLayerDescs[0]._qAlpha = aLayerDescs[layer][i]._qAlpha * lrScalar;
					agentVisibleLayerDescs[0]._actionAlpha = aLayerDescs[layer][i]._actionAlpha * lrScalar;

					const int2 size = _p.getHierarchy().getLayer(layer)._sf->getHiddenSize();
					agentVisibleLayerDescs[0]._size = (layer == 0) ? size : int2{ size.x * 2, size.y * 2 };
					_aLayers[layer][i].createRandom(
						(layer == _aLayers.size() - 1) 
							? actionSizes[i] 
							: _p.getHierarchy().getLayer(layer + 1)._sf->getHiddenSize(), 
						(layer == _aLayers.size() - 1) 
							? actionTileSizes[i] 
							: int2{ 2, 2 }, 
						agentVisibleLayerDescs, 
						initWeightRange, 
						rng);
				}
			}

			_ones.resize(_aLayers.back().size());

			for (size_t i = 0; i < _ones.size(); ++i) 
			{
				_ones[i] = Array2D2f(actionSizes[i]);
				_ones[i].fill(float2{ 1.0f, 1.0f });
			}

			_rewardSums.clear();
			_rewardSums.assign(_aLayers.size(), 0.0f);

			_rewardCounts.clear();
			_rewardCounts.assign(_aLayers.size(), 0.0f);
		}

		/*!
		\brief Simulation step of hierarchy
		Takes reward and inputs, optionally disable learning.
		\param reward the reinforcement learning signal.
		\param input the input layer state.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const float reward,
			const std::vector<Array2D2f> &inputs,
			const std::vector<Array2D2f> &inputsCorrupted,
			std::mt19937 &rng,
			const bool learn = true) 
		{
			// Activate hierarchy
			_p.simStep(inputs, inputsCorrupted, rng, learn);

			// Update agent layers
			for (size_t layer = 0; layer < _aLayers.size(); layer++)
			{
				_rewardSums[layer] += reward;
				_rewardCounts[layer] += 1.0f;
				float totalReward = _rewardSums[layer] / _rewardCounts[layer];

				for (size_t i = 0; i < _aLayers[layer].size(); i++) {
					Array2D2f inputs = (layer == 0)
						? _p.getHierarchy().getLayer(layer)._sf->getHiddenStates()[_back] 
						: _aLayers[layer - 1].front().getOneHotActions();

					if (layer == _aLayers.size() - 1) {
						_aLayers[layer][i].simStep(
							totalReward, 
							std::vector<Array2D2f>(1, inputs),
							_ones[i], 
							_aLayerDescs[layer][i]._qGamma,
							_aLayerDescs[layer][i]._qLambda, 
							_aLayerDescs[layer][i]._actionLambda, 
							_aLayerDescs[layer][i]._maxActionWeightMag, 
							rng, 
							learn);
					} 
					else {
						_aLayers[layer][i].simStep(
							totalReward, 
							std::vector<Array2D2f>(1, inputs),
							_p.getHierarchy().getLayer(layer + 1)._sf->getHiddenStates()[_back], 
							_aLayerDescs[layer][i]._qGamma, 
							_aLayerDescs[layer][i]._qLambda,
							_aLayerDescs[layer][i]._actionLambda,
							_aLayerDescs[layer][i]._maxActionWeightMag,
							rng, 
							learn);
					}
				}

				_rewardSums[layer] = 0.0f;
				_rewardCounts[layer] = 0.0f;
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
		const std::vector<AgentLayer> &getAgentLayer(int index) const {
			return _aLayers[index];
		}

		/*!
		\brief Get access to an agent layer descriptor
		*/
		const std::vector<AgentLayerDesc> &getAgentLayerDesc(int index) const {
			return _aLayerDescs[index];
		}

		/*!
		\brief Get the actions
		Returns float 2D image where each element is actually an integer, representing the index of the select action for each tile.
		To get continuous values, divide each tile index by the number of elements in a tile (actionTileSize.x * actionTileSize.y).
		*/
		const Image2D &getAction(int index) const {
			return _aLayers.back()[index].getActions()[_back];
		}

		//Get the underlying feature hierarchy
		Predictor &getPredictor() {
			return _p;
		}
	};
}