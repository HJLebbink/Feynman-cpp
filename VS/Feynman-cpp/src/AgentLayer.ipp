// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <tuple>
#include <vector>
#include <random>

#include "Helpers.ipp"
#include "SparseFeatures.ipp"

namespace feynman {

	//Agent layer.
	//Contains a (2D) swarm of small Q learning agents, one per action tile
	class AgentLayer {

	public:

		//Layer desc for inputs to the swarm layer
		struct VisibleLayerDesc {

			//Layer properties: Size, radius onto layer, and learning rate
			int2 _size;
			int _radius;
			float _alpha;
			
			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 16, 16 }),
				_radius(10),
				_alpha(0.01f)
			{}
		};

		//Layer
		struct VisibleLayer {
			//Layer data
			DoubleBuffer3D _weights;

			float2 _hiddenToVisible;
			float2 _visibleToHidden;

			int2 _reverseRadii;
		};

	private:

		//Size of action layer in tiles
		int2 _numActionTiles;

		//Size of an action tile
		int2 _actionTileSize;

		//Size of the total action region (hidden size), (numActionTiles.x * actionTileSize.x, numActionTiles.y * actionTileSize.y)
		int2 _hiddenSize;

		//Hidden state variables: Q states, actions, td errors, one hot action
		DoubleBuffer2D _qStates;

		DoubleBuffer2D _action;
		DoubleBuffer2D _actionTaken;
		Image2D _tdError;
		Image2D _oneHotAction;

		//Hidden stimulus summation temporary buffer
		DoubleBuffer2D _hiddenSummationTemp;

		//Layers and descs
		std::vector<VisibleLayer> _visibleLayers;
		std::vector<VisibleLayerDesc> _visibleLayerDescs;

	public:
		//Initialize defaults
		AgentLayer()
		{}

		/*!
		\brief Create a predictive hierarchy with random initialization.
		Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
		\param cs is the ComputeSystem.
		\param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
		\param numActionTiles is the (2D) size of the action layer.
		\param actionTileSize is the (2D) size of each action tile (square one-hot action region).
		\param visibleLayerDescs is a vector of visible layer parameters.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const int2 numActionTiles,
			const int2 actionTileSize,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const float2 initWeightRange,
			const std::mt19937 &rng) 
		{
			_visibleLayerDescs = visibleLayerDescs;
			_numActionTiles = numActionTiles;
			_actionTileSize = actionTileSize;
			_hiddenSize = { _numActionTiles.x * _actionTileSize.x, _numActionTiles.y * _actionTileSize.y };
			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (int vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				vl._hiddenToVisible = float2{ 
					static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
					static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
				};
				vl._visibleToHidden = float2{ 
					static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
					static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
				};
				vl._reverseRadii = int2{ 
					static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
				};
				{
					int weightDiam = vld._radius * 2 + 1;
					int numWeights = weightDiam * weightDiam;
					int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._weights = createDoubleBuffer3D(weightsSize);
					randomUniform3D<(vl._weights[_back], weightsSize, initWeightRange, rng);
				}
			}

			// Hidden state data
			_qStates = createDoubleBuffer2D(_hiddenSize);
			_action = createDoubleBuffer2D(_numActionTiles);
			_actionTaken = createDoubleBuffer2D(_numActionTiles);

			_tdError = Image2D(_hiddenSize);
			_oneHotAction = Image2D(_hiddenSize.x);

			clear(_qStates[_back]);
			clear(_action[_back]);
			clear(_actionTaken[_back]);

			_hiddenSummationTemp = createDoubleBuffer2D(_hiddenSize);
		}

		/*!
		\brief Simulation step of agent layer agents.
		Requres several reinforcement learning parameters.
		\param reward the reinforcement learning signal.
		\param visibleStates all input layer states.
		\param modulator layer that modulates the agents in the swarm (1 = active, 0 = inactive).
		\param qGamma Q learning gamma.
		\param qLambda Q learning lambda (trace decay).
		\param epsilon Q learning epsilon greedy exploration rate.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const float reward, 
			const std::vector<Image2D> &visibleStates,
			const Image2D &modulator,
			const float qGamma,
			const float qLambda,
			const float epsilon,
			const std::mt19937 &rng,
			const bool learn = true)
		{
			clear(_hiddenSummationTemp[_back]);

			// Find Q
			for (int vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				alFindQ(
					visibleStates[vli],
					vl._weights[_back],
					_hiddenSummationTemp[_back],
					_hiddenSummationTemp[_front],
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					_hiddenSize);

				// Swap buffers
				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}

			// Copy to hidden states
			copy(_hiddenSummationTemp[_back], _qStates[_front]);

			// Get newest actions
			alGetAction(
				_qStates[_front],
				_action[_front],
				_actionTileSize,
				_numActionTiles);

			std::swap(_action[_front], _action[_back]);

			// Exploration
			{
				std::uniform_int_distribution<int> seedDist(0, 9999);
				uint2 seed = { static_cast<uint>(seedDist(rng)),static_cast<uint>(seedDist(rng)) };

				alActionExploration(
					_action[_back],
					_actionTaken[_front],
					epsilon,
					_actionTileSize.x * _actionTileSize.y,
					seed,
					_numActionTiles);

				std::swap(_actionTaken[_front], _actionTaken[_back]);
			}

			// Compute TD errors
			alSetAction(
				modulator,
				_action[_back],
				_action[_front],
				_actionTaken[_back],
				_actionTaken[_front],
				_qStates[_front],
				_qStates[_back],
				_tdError,
				_oneHotAction,
				_actionTileSize,
				reward,
				qGamma,
				_numActionTiles);

			std::swap(_qStates[_front], _qStates[_back]);

			if (learn) {
				for (int vli = 0; vli < _visibleLayers.size(); vli++) {
					VisibleLayer &vl = _visibleLayers[vli];
					VisibleLayerDesc &vld = _visibleLayerDescs[vli];

					// Learn Q
					alLearnQ(
						visibleStates[vli],
						_qStates[_back],
						_qStates[_front],
						_tdError,
						_oneHotAction,
						vl._weights[_back],
						vl._weights[_front],
						vld._size,
						vl._hiddenToVisible,
						vld._radius,
						vld._alpha,
						qLambda,
						_hiddenSize);

					std::swap(vl._weights[_front], vl._weights[_back]);
				}
			}
		}

		//Clear memory (recurrent data)
		void clearMemory() 
		{
			clear(_qStates[_back]);
			clear(_action[_back]);
			clear(_actionTaken[_back]);
		}

		//Get number of layers
		size_t getNumLayers() const {
			return _visibleLayers.size();
		}

		/*!
		\brief Get access to a layer
		\param[in] index Visible layer index.
		*/
		const VisibleLayer &getLayer(int index) const {
			return _visibleLayers[index];
		}

		/*!
		\brief Get access to a layer descriptor
		\param[in] index Visible layer descriptor index.
		*/
		const VisibleLayerDesc &getLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		//Get the Q states
		const DoubleBuffer2D &getQStates() const {
			return _qStates;
		}

		//Get the actions
		const DoubleBuffer2D &getActions() const {
			return _actionTaken;
		}

		//Get the actions in one-hot form
		const Image2D &getOneHotActions() const {
			return _oneHotAction;
		}

		//Get number of action tiles in X and Y
		int2 getNumActionTiles() const {
			return _numActionTiles;
		}

		//Get size of action tiles in X and Y
		int2 getActionTileSize() const {
			return _actionTileSize;
		}

		//Get the hidden size
		int2 getHiddenSize() const {
			return _hiddenSize;
		}

		private:

		static void alFindQ(
			const Image2D &hiddenStates,
			const Image3D &weights,
			const Image2D &hiddenSummationBack,
			Image2D &hiddenSummationFront,
			const int2 hiddenSize,
			const float2 qToHidden,
			const int radius,
			const int2 range)
		{
			int2 qPosition;
			for (int x = 0; x < range.x; ++x) {
				qPosition.x = x;
				for (int y = 0; y < range.y; ++y) {
					qPosition.y = y;
					int2 hiddenPositionCenter = project(qPosition, qToHidden);
					float sum = read_2D(hiddenSummationBack, qPosition);
					float q = 0.0f;
					int2 fieldLowerBound = hiddenPositionCenter - int2{ radius };
					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							int2 hiddenPosition = hiddenPositionCenter + int2{ dx, dy };
							if (inBounds(hiddenPosition, hiddenSize)) {
								int2 offset = hiddenPosition - fieldLowerBound;
								int wi = offset.y + offset.x * (radius * 2 + 1);
								float weight = read_3D(weights, qPosition.x, qPosition.y, wi);
								float state = read_2D(hiddenStates, hiddenPosition);
								q += state * weight;
							}
						}
					}
					write_2D(hiddenSummationFront, x, y, sum + q);
				}
			}
		}

		static void alLearnQ(
			const Image2D &hiddenStates,
			const Image2D<float2> &qStates,
			const Image2D<float2> &qStatesPrev,
			const Image2D &tdErrors,
			const Image2D &oneHotActions,
			const Image3D<float2> &weightsBack,
			Image3D<float2> &weightsFront,
			const int2 hiddenSize,
			const float2 qToHidden,
			const int radius,
			const float alpha,
			const float lambda,
			const int2 range)
		{
			int2 qPosition;
			for (int x = 0; x < range.x; ++x) {
				qPosition.x = x;
				for (int y = 0; y < range.y; ++y) {
					qPosition.y = y;

					int2 hiddenPositionCenter = project(qPosition, qToHidden);

					float tdError = read_2D(tdErrors, qPosition);
					float oneHotAction = read_2D(oneHotActions, qPosition);
					//float2 qState = read_imagef_2D_2x(qStates, qPosition); //TODO: investigate
					//float2 qStatePrev = read_imagef_2D_2x(qStatesPrev, qPosition); //TODO: investigate

					int2 fieldLowerBound = hiddenPositionCenter - int2{ radius };

					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							int2 hiddenPosition = hiddenPositionCenter + int2{ dx, dy };

							if (inBounds(hiddenPosition, hiddenSize)) {
								int2 offset = hiddenPosition - fieldLowerBound;
								int wi = offset.y + offset.x * (radius * 2 + 1);
								float2 weightPrev = read_3D(weightsBack, qPosition.x, qPosition.y, wi);
								float state = read_2D(hiddenStates, hiddenPosition);
								float2 weight = float2{ weightPrev.x + alpha * tdError * weightPrev.y, lambda * weightPrev.y + (1.0f - lambda) * oneHotAction * state };
								write_3D(weightsFront, qPosition.x, qPosition.y, wi, weight);
							}
						}
					}
				}
			}
		}

		static void alActionToOneHot(
			const Image2D &hiddenStates,
			const Image2D &actions,
			Image2D &oneHotActions,
			const int2 subActionDims,
			const uchar modulate,
			const int2 range)
		{
			int2 position;
			for (int x = 0; x < range.x; ++x) {
				position.x = x;
				for (int y = 0; y < range.y; ++y) {
					position.y = y;

					float hiddenState = modulate ? read_2D(hiddenStates, x, y): 1.0f;
					float action = read_2D(actions, x, y);
					int actioni = (int)(round(action));
					int2 actionPosition = position * subActionDims + int2{ actioni % subActionDims.x, actioni / subActionDims.x };
					for (int x = 0; x < subActionDims.x; x++) {
						for (int y = 0; y < subActionDims.y; y++) {
							int index = x + y * subActionDims.x;
							int2 subPosition = position * subActionDims + int2{ x, y };
							write_2D(oneHotActions, subPosition, (index == actioni) ? hiddenState : 0.0f);
						}
					}
				}
			}
		}

		static void alGetAction(
			const Image2D &predictions,
			Image2D &actions,
			const int2 subActionDims,
			const int2 range)
		{
			int2 position;
			for (int x = 0; x < range.x; ++x) {
				position.x = x;
				for (int y = 0; y < range.y; ++y) {
					position.y = y;
					int maxIndex = 0;
					float maxValue = -99999.0f;

					for (int x = 0; x < subActionDims.x; x++) {
						for (int y = 0; y < subActionDims.y; y++) {
							float value = read_2D(predictions, position * subActionDims + int2{ x, y });

							if (value > maxValue) {
								maxValue = value;
								maxIndex = x + y * subActionDims.x;
							}
						}
					}
					write_2D(actions, x, y, float4{ maxIndex });
				}
			}
		}

		static void alSetAction(
			const Image2D &modulator,
			const Image2D &actions,
			const Image2D &actionsPrev,
			const Image2D &actionsTaken,
			const Image2D &actionsTakenPrev,
			const Image2D &predictions,
			const Image2D &predictionsPrev,
			Image2D &tdErrorsTrain,
			Image2D &oneHotActions,
			const int2 subActionDims,
			const float reward,
			const float gamma,
			const int2 range)
		{
			int2 position;
			for (int x = 0; x < range.x; ++x) {
				position.x = x;
				for (int y = 0; y < range.y; ++y) {
					position.y = y;
					float modulate = read_2D(modulator, x, y);

					float action = read_2D(actions, x, y);
					float actionPrev = read_2D(actionsPrev, x, y);
					float actionTaken = read_2D(actionsTaken, x, y);
					float actionTakenPrev = read_2D(actionsTakenPrev, x, y);

					int actioni = (int)(round(action));
					int actionPrevi = (int)(round(actionPrev));
					int actionTakeni = (int)(round(actionTaken));
					int actionTakenPrevi = (int)(round(actionTakenPrev));

					int2 actionPosition = position * subActionDims + int2{ actioni % subActionDims.x, actioni / subActionDims.x };
					int2 actionPrevPosition = position * subActionDims + int2{ actionPrevi % subActionDims.x, actionPrevi / subActionDims.x };
					int2 actionTakenPosition = position * subActionDims + int2{ actionTakeni % subActionDims.x, actionTakeni / subActionDims.x };
					int2 actionTakenPrevPosition = position * subActionDims + int2{ actionTakenPrevi % subActionDims.x, actionTakenPrevi / subActionDims.x };

					float pred = read_2D(predictions, actionTakenPosition);
					float predPrev = read_2D(predictionsPrev, actionTakenPrevPosition);

					float tdError = reward + gamma * pred - predPrev;

					for (int x = 0; x < subActionDims.x; x++)
						for (int y = 0; y < subActionDims.y; y++) {
							int index = x + y * subActionDims.x;

							int2 subPosition = position * subActionDims + int2{ x, y };

							write_2D(tdErrorsTrain, subPosition, float4{ tdError });
							write_2D(oneHotActions, subPosition, float4{ index == actionTakeni ? modulate : 0.0f });
						}
				}
			}
		}

		static void alActionExploration(
			const Image2D &actions,
			Image2D &actionsExploratory,
			const float epsilon,
			const int subActionCount,
			const uint2 seed,
			const int2 range)
		{
			uint2 seedValue;
			int2 position;
			for (int x = 0; x < range.x; ++x) {
				seedValue.x = ((x * 12) + 43) * 12;
				position.x = x;
				for (int y = 0; y < range.y; ++y) {
					seedValue.y = ((y * 21) + 42) * 12;
					position.y = y;

					int aexi;
					if (randFloat(&seedValue) < epsilon) {
						// Exploratory action
						aexi = static_cast<int>(randFloat(&seedValue) * subActionCount);
					}
					else {
						aexi = static_cast<int>(round(read_2D(actions, x, y)));
					}
					write_2D(actionsExploratory, x, y, static_cast<float>(aexi));
				}
			}
		}
	};
}