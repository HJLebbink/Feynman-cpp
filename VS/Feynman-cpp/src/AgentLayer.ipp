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
			float _qAlpha;
			float _actionAlpha;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 16, 16 }),
				_radius(12),
				_qAlpha(0.001f),
				_actionAlpha(0.02f)
			{}
		};

		//Layer
		struct VisibleLayer {
			//Layer data
			DoubleBuffer3D _qWeights;
			DoubleBuffer3D _actionWeights;

			float2 _qToVisible;
			float2 _visibleToQ;

			int2 _reverseRadiiQ;

			float2 _hiddenToVisible;
			float2 _visibleToHidden;

			int2 _reverseRadiiHidden;
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
		DoubleBuffer2D _actionProbabilities;

		DoubleBuffer2D _actionTaken;
		Image2D _tdError;
		Image2D _oneHotAction;

		//Hidden stimulus summation temporary buffer
		DoubleBuffer2D _hiddenSummationTempQ;
		DoubleBuffer2D _hiddenSummationTempHidden;

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
			std::mt19937 &rng) 
		{
			_visibleLayerDescs = visibleLayerDescs;
			_numActionTiles = numActionTiles;
			_actionTileSize = actionTileSize;
			_hiddenSize = { _numActionTiles.x * _actionTileSize.x, _numActionTiles.y * _actionTileSize.y };
			_visibleLayers.resize(_visibleLayerDescs.size());

			// Create layers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				vl._qToVisible = float2{ 
					static_cast<float>(vld._size.x) / static_cast<float>(_numActionTiles.x),
					static_cast<float>(vld._size.y) / static_cast<float>(_numActionTiles.y)
				};
				vl._visibleToQ = float2{
					static_cast<float>(_numActionTiles.x) / static_cast<float>(vld._size.x),
					static_cast<float>(_numActionTiles.y) / static_cast<float>(vld._size.y)
				};
				vl._reverseRadiiQ = int2{ 
					static_cast<int>(std::ceil(vl._visibleToQ.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToQ.y * vld._radius) + 1)
				};
				vl._hiddenToVisible = float2{
					static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
					static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
				};
				vl._visibleToHidden = float2{
					static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
					static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
				};
				vl._reverseRadiiHidden = int2{ 
					static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
				};
				{
					const int weightDiam = vld._radius * 2 + 1;
					const int numWeights = weightDiam * weightDiam;
					const int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._qWeights = createDoubleBuffer3D(weightsSize);
					randomUniform3D(vl._qWeights[_back], weightsSize, initWeightRange, rng);
				}
			}

			// Hidden state data
			_qStates = createDoubleBuffer2D(_hiddenSize);
			_actionProbabilities = createDoubleBuffer2D(_hiddenSize);
			_actionTaken = createDoubleBuffer2D(_numActionTiles);

			_tdError = Image2D(_numActionTiles);
			_oneHotAction = Image2D(_hiddenSize);

			clear(_qStates[_back]);
			clear(_actionProbabilities[_back]);
			clear(_actionTaken[_back]);

			_hiddenSummationTempQ = createDoubleBuffer2D(_numActionTiles);
			_hiddenSummationTempHidden = createDoubleBuffer2D(_hiddenSize);
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
			const float actionLambda,
			const float maxActionWeightMag,
			std::mt19937 &rng,
			const bool learn)
		{
			clear(_hiddenSummationTempQ[_back]);
			clear(_hiddenSummationTempHidden[_back]);

			// Find Q
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				alActivate(
					visibleStates[vli],				// in
					vl._qWeights[_back],			// in
					_hiddenSummationTempQ[_back],	// in
					_hiddenSummationTempQ[_front],	// out
					vld._size,
					vl._qToVisible,
					vld._radius, 
					_numActionTiles);

				// Swap buffers
				std::swap(_hiddenSummationTempQ[_front], _hiddenSummationTempQ[_back]);

				alActivate(
					visibleStates[vli],				// in
					vl._actionWeights[_back],		// in
					_hiddenSummationTempHidden[_back],	// in
					_hiddenSummationTempHidden[_front],	// out
					vld._size,
					vl._hiddenToVisible,
					vld._radius,
					_hiddenSize);

				// Swap buffers
				std::swap(_hiddenSummationTempHidden[_front], _hiddenSummationTempHidden[_back]);
			}

			// Copy to Q states
			copy(_hiddenSummationTempQ[_back], _qStates[_front]);

			// Get newest actions
			std::uniform_int_distribution<int> seedDist(0, 9999);
			uint2 seed = { static_cast<unsigned int>(seedDist(rng)), static_cast<unsigned int>(seedDist(rng)) };

			alGetAction(
				_hiddenSummationTempHidden[_back],	// in
				_actionProbabilities[_front],		// in
				_actionTaken[_front],				// out
				_actionTileSize,
				seed,
				_numActionTiles);

			std::swap(_actionTaken[_front], _actionTaken[_back]);

			// Compute TD errors
			alSetAction(
				modulator,				// in
				_actionTaken[_back],	// in
				_actionTaken[_front],	// in
				_qStates[_back],		// in
				_qStates[_front],		// in
				_tdError,				// in
				_oneHotAction,			// out
				_actionTileSize,		// out
				reward,
				qGamma,
				_numActionTiles);

			std::swap(_qStates[_front], _qStates[_back]);
			std::swap(_actionProbabilities[_front], _actionProbabilities[_back]);

			if (learn) {
				for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
					VisibleLayer &vl = _visibleLayers[vli];
					VisibleLayerDesc &vld = _visibleLayerDescs[vli];

					// Learn Q
					alLearnQ(
						visibleStates[vli],		// in
						_qStates[_back],		// in // unused
						_qStates[_front],		// in // unused
						_tdError,				// in
						_oneHotAction,			// in
						vl._qWeights[_back],		// in
						vl._qWeights[_front],	// out
						vld._size,
						vl._hiddenToVisible,
						vld._radius,
						vld._qAlpha,
						qLambda,
						_numActionTiles);

					std::swap(vl._qWeights[_front], vl._qWeights[_back]);

					// Learn action
					alLearnActions(
						visibleStates[vli],				// in
						_actionProbabilities[_front],	// in // unused
						_tdError,						// in // unused
						_oneHotAction,				// in
						vl._actionWeights[_back],			// in
						vl._actionWeights[_front],		// in
						vld._size,
						vl._hiddenToVisible,
						vld._radius,
						vld._actionAlpha,
						actionLambda,
						_actionTileSize,
						maxActionWeightMag,
						_hiddenSize);

					std::swap(vl._actionWeights[_front], vl._actionWeights[_back]);

				}
			}
		}

		//Clear memory (recurrent data)
		void clearMemory() 
		{
			clear(_qStates[_back]);
			clear(_actionProbabilities[_back]);
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

		static void alActivate(
			const Image2D &hiddenStates,
			const Image3D &weights,
			const Image2D &hiddenSummationBack,
			Image2D &hiddenSummationFront,
			const int2 hiddenSize,
			const float2 qToHidden,
			const int radius,
			const int2 range)
		{
			printf("AgentLayer::alActivate: TODO");
			/*
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
			*/
		}

		static void alLearnQ(
			const Image2D &hiddenStates,
			const Image2D &qStates,	// unused
			const Image2D &qStatesPrev,	//unused
			const Image2D &tdErrors,
			const Image2D &oneHotActions,
			const Image3D &weightsBack,
			Image3D &weightsFront,
			const int2 hiddenSize,
			const float2 qToHidden,
			const int radius,
			const float alpha,
			const float lambda,
			const int2 range)
		{
			for (int x = 0; x < range.x; ++x) {
				for (int y = 0; y < range.y; ++y) {

					const int hiddenPositionCenter_x = project(x, qToHidden.x);
					const int hiddenPositionCenter_y = project(y, qToHidden.y);

					const float tdError = read_2D(tdErrors, x, y);
					const float oneHotAction = read_2D(oneHotActions, x, y);

					//float2 qState = read_imagef_2D_2x(qStates, qPosition); //TODO: investigate
					//float2 qStatePrev = read_imagef_2D_2x(qStatesPrev, qPosition); //TODO: investigate

					const int fieldLowerBound_x = hiddenPositionCenter_x - radius;
					const int fieldLowerBound_y = hiddenPositionCenter_y - radius;

					for (int dx = -radius; dx <= radius; ++dx) {
						const int hiddenPosition_x = hiddenPositionCenter_x + dx;
						const int offset_x = hiddenPosition_x - fieldLowerBound_x;

						for (int dy = -radius; dy <= radius; ++dy) {
							const int hiddenPosition_y = hiddenPositionCenter_y + dy;

							if (inBounds(hiddenPosition_x, hiddenSize.x) && inBounds(hiddenPosition_y, hiddenSize.y)) {
								const int offset_y = hiddenPosition_y - fieldLowerBound_y;

								const int wi = offset_y + (offset_x * ((radius * 2) + 1));

								const float weightPrev1 = read_3D(weightsBack, x, y, wi);
								const float weightPrev2 = 0;// TODO read_3D(weightsBack, x, y, wi);
								printf("AgentLayer::TODO");

								const float state = read_2D(hiddenStates, hiddenPosition_x, hiddenPosition_y);
								const float weight1 = weightPrev1 + alpha * tdError * weightPrev2;
								const float weight2 = lambda * weightPrev2 + (1.0f - lambda) * oneHotAction * state;
								write_3D(weightsFront, x, y, wi, weight1);
							}
						}
					}
				}
			}
		}

		static void alLearnActions(
			const Image2D &hiddenStates, 
			const Image2D &actionProbabilities,
			const Image2D &tdErrors, 
			const Image2D &oneHotActions,
			const Image3D &weightsBack, 
			Image3D &weightsFront, // write only
			const int2 hiddenSize, 
			const float2 aToHidden, 
			const int radius, 
			const float alpha, 
			const float lambda, 
			const int2 subActionDims, 
			const float maxActionWeightMag,
			const int2 range)
		{
			printf("AgentLayer::alLearnActions: TODO");

			/*
			for (int x = 0; x < range.x; ++x) {
				for (int y = 0; y < range.y; ++y) {
					const int hiddenPositionCenter_x = project(x, aToHidden.x);
					const int hiddenPositionCenter_y = project(y, aToHidden.y);

					float tdError = read_imagef(tdErrors, defaultSampler, (int2)(aPosition.x / subActionDims.x, aPosition.y / subActionDims.y)).x;
					float2 oneHotAction = read_imagef(oneHotActions, defaultSampler, aPosition).xy;

					float probability = read_imagef(actionProbabilities, defaultSampler, aPosition).x;

					int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

							if (inBounds0(hiddenPosition, hiddenSize)) {
								int2 offset = hiddenPosition - fieldLowerBound;
								int wi = offset.y + offset.x * (radius * 2 + 1);
								float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(aPosition.x, aPosition.y, wi, 0)).xy;
								float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
								float2 weight = (float2)(weightPrev.x + alpha * tdError * weightPrev.y, lambda * weightPrev.y + oneHotAction.y * (1.0f - lambda) * (oneHotAction.x - probability) * state);
								write_imagef(weightsFront, (int4)(aPosition.x, aPosition.y, wi, 0), (float4)(fmin(maxActionWeightMag, fmax(-maxActionWeightMag, weight.x)), weight.y, 0.0f, 0.0f));
							}
						}
					}
				}
			}
			*/
		}

		static void alActionToOneHot(
			const Image2D &hiddenStates,
			const Image2D &actions,
			Image2D &oneHotActions,
			const int2 subActionDims,
			const bool modulate,
			const int2 range)
		{
			for (int x = 0; x < range.x; ++x) {
				for (int y = 0; y < range.y; ++y) {

					const float hiddenState = modulate ? read_2D(hiddenStates, x, y): 1.0f;
					const float action = read_2D(actions, x, y);
					const int actioni = static_cast<int>(round(action));

					const int actionPosition_x = x * subActionDims.x + actioni % subActionDims.x;
					const int actionPosition_y = y * subActionDims.y + actioni / subActionDims.x;
					
					for (int x = 0; x < subActionDims.x; x++) {
						for (int y = 0; y < subActionDims.y; y++) {
							const int index = x + (y * subActionDims.x);
							const int subPosition_x = x * subActionDims.x + x;
							const int subPosition_y = y * subActionDims.y + y;
							write_2D(oneHotActions, subPosition_x, subPosition_y, (index == actioni) ? hiddenState : 0.0f);
						}
					}
				}
			}
		}

		static void alGetAction(
			const Image2D &activations,
			Image2D &probabilities,
			const Image2D &actions,
			const int2 subActionDims,
			const uint2 seed,
			const int2 range)
		{
			printf("AgentLayer::alSetAction: TODO");

			for (int x = 0; x < range.x; ++x) {
				const unsigned int seedValue_x = seed.x + (x * 73 + 2) * 44;

				for (int y = 0; y < range.y; ++y) {
					const unsigned int seedValue_y = seed.y + (y * 45 + 12) * 44;

					float expSum = 0.0f;

					/*


					for (int x = 0; x < subActionDims.x; x++)
						for (int y = 0; y < subActionDims.y; y++) {
							float value = read_imagef(activations, defaultSampler, position * subActionDims + (int2)(x, y)).x;

							expSum += exp(value);
						}

					float select = randFloat(&seedValue);

					int selectIndex = 0;

					float sumSoFar = 0.0f;

					uchar selected = 0;

					for (int x = 0; x < subActionDims.x; x++)
						for (int y = 0; y < subActionDims.y; y++) {
							int2 subPosition = position * subActionDims + (int2)(x, y);

							float value = read_imagef(activations, defaultSampler, subPosition).x;

							float probability = exp(value) / expSum;

							write_imagef(probabilities, subPosition, (float4)(probability, 0.0f, 0.0f, 0.0f));

							sumSoFar += probability;

							if (!selected && sumSoFar >= select) {
								selectIndex = x + y * subActionDims.x;

								selected = 1;
							}
						}

					write_imagef(actions, position, (float4)(selectIndex));
					*/
				}
			}
		}

		static void alSetAction(
			const Image2D &modulator,
			const Image2D &actionsTaken,
			const Image2D &actionsTakenPrev,
			const Image2D &qStates,
			const Image2D &qStatesPrev,
			Image2D &tdErrorsTrain,
			Image2D &oneHotActions,
			const int2 subActionDims,
			const float reward,
			const float gamma,
			const int2 range)
		{
			printf("AgentLayer::alSetAction: TODO");
/*
			for (int x = 0; x < range.x; ++x) {
				for (int y = 0; y < range.y; ++y) {
					const float modulate = read_2D(modulator, x, y);

					const float action = read_2D(actions, x, y);
					const float actionPrev = read_2D(actionsPrev, x, y);
					const float actionTaken = read_2D(actionsTaken, x, y);
					const float actionTakenPrev = read_2D(actionsTakenPrev, x, y);

					const int actioni = static_cast<int>(round(action));
					const int actionPrevi = static_cast<int>(round(actionPrev));
					const int actionTakeni = static_cast<int>(round(actionTaken));
					const int actionTakenPrevi = static_cast<int>(round(actionTakenPrev));

					const int actionPosition_x = x * subActionDims.x + actioni % subActionDims.x;
					const int actionPosition_y = y * subActionDims.y + actioni / subActionDims.x;
					const int actionPrevPosition_x = x * subActionDims.x + actionPrevi % subActionDims.x;
					const int actionPrevPosition_y = y * subActionDims.y + actionPrevi / subActionDims.x;
					const int actionTakenPosition_x = x * subActionDims.x + actionTakeni % subActionDims.x;
					const int actionTakenPosition_y = y * subActionDims.y + actionTakeni / subActionDims.x;
					const int actionTakenPrevPosition_x = x * subActionDims.x + actionTakenPrevi % subActionDims.x;
					const int actionTakenPrevPosition_y = y * subActionDims.y + actionTakenPrevi / subActionDims.x;

					const float pred = read_2D(predictions, actionTakenPosition_x, actionTakenPosition_y);
					const float predPrev = read_2D(predictionsPrev, actionTakenPrevPosition_x, actionTakenPrevPosition_y);
					const float tdError = reward + gamma * pred - predPrev;

					for (int x = 0; x < subActionDims.x; x++) {
						for (int y = 0; y < subActionDims.y; y++) {
							const int index = x + y * subActionDims.x;
							const int subPosition_x = x * subActionDims.x + x;
							const int subPosition_y = y * subActionDims.y + y;
							write_2D(tdErrorsTrain, subPosition_x, subPosition_y, tdError);
							write_2D(oneHotActions, subPosition_x, subPosition_y, (index == actionTakeni) ? modulate : 0.0f);
						}
					}
				}
			}
			*/
		}
	};
}