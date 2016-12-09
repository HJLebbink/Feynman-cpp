// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once
#include <unordered_map>
#include <sstream>
#include <memory>

#include "Agent.ipp"
#include "Hierarchy.ipp"

// Encoders
#include "SparseFeaturesOld.ipp"
#include "SparseFeaturesChunk.ipp"
#include "SparseFeaturesDelay.ipp"
#include "SparseFeaturesSTDP.ipp"

#include <iostream>

namespace feynman {

	static const std::string _boolTrue = "true";
	static const std::string _boolFalse = "false";

	//Parameter modified interface
	class ParameterModifier {
	public:

	private:
		std::unordered_map<std::string, std::string>* _target;

	public:

		ParameterModifier &setValue(const std::string &name, const std::string &value) {
			(*_target)[name] = value;
			return *this;
		}

		ParameterModifier &setValueBool(const std::string &name, bool value) {
			(*_target)[name] = value ? _boolTrue : _boolFalse;
			return *this;
		}

		ParameterModifier &setValue(const std::string &name, float value) {
			(*_target)[name] = std::to_string(value);
			return *this;
		}

		ParameterModifier &setValue(const std::string &name, const int2 &size) {
			(*_target)[name] = "(" + std::to_string(size.x) + ", " + std::to_string(size.y) + ")";
			return *this;
		}

		ParameterModifier &setValue(const std::string &name, const float2 &size) {
			(*_target)[name] = "(" + std::to_string(size.x) + ", " + std::to_string(size.y) + ")";
			return *this;
		}

		ParameterModifier &setValues(const std::vector<std::pair<std::string, std::string>> &namesValues) {
			for (size_t i = 0; i < namesValues.size(); i++) {
				setValue(std::get<0>(namesValues[i]), std::get<1>(namesValues[i]));
			}
			return *this;
		}

		static int2 parseInt2(const std::string &s) {
			std::istringstream is(s.substr(1, s.size() - 2)); // Remove ()
			std::string xs;
			std::getline(is, xs, ',');
			std::string ys;
			std::getline(is, ys);
			return int2{ std::stoi(xs), std::stoi(ys) };
		}

		static float2 parseFloat2(const std::string &s) {
			std::istringstream is(s.substr(1, s.size() - 2)); // Remove ()
			std::string xs;
			std::getline(is, xs, ',');
			std::string ys;
			std::getline(is, ys);
			return float2{ std::stof(xs), std::stof(ys) };
		}

		static bool parseBool(const std::string &s) {
			return s == _boolTrue;
		}

		friend class Architect;
	};

	struct InputLayer {
		int2 _size;
		std::unordered_map<std::string, std::string> _params;
	};

	struct ActionLayer {
		int2 _size;
		int2 _tileSize;
		std::unordered_map<std::string, std::string> _params;
	};

	struct HigherLayer {
		int2 _size;
		SparseFeaturesType _type;
		std::unordered_map<std::string, std::string> _params;
	};

	/*!
	\brief Hierarchy architect
	Used to create hierarchies with a simple interface. Generates an agent or a hierarchy based on the specifications provided.
	*/
	class Architect {
	private:

		std::vector<InputLayer> _inputLayers;
		std::vector<ActionLayer> _actionLayers;
		std::vector<HigherLayer> _higherLayers;

		std::mt19937 _rng;

		std::shared_ptr<SparseFeatures::SparseFeaturesDesc> sfDescFromName(
			const int layerIndex,
			const SparseFeaturesType type,
			const int2 size,
			const SparseFeatures::InputType inputType,
			std::unordered_map<std::string, std::string> &params)
		{
			std::shared_ptr<SparseFeatures::SparseFeaturesDesc> sfDesc;

			switch (type) {
			case _old: 
			{
				std::shared_ptr<SparseFeaturesOld::SparseFeaturesOldDesc> sfDesc_Old = std::make_shared<SparseFeaturesOld::SparseFeaturesOldDesc>();

				sfDesc_Old->_inputType = SparseFeatures::_feedForwardRecurrent;
				sfDesc_Old->_hiddenSize = size;
				sfDesc_Old->_rng = _rng;

				if (params.find("inhibitionRadius") != params.end())
					sfDesc_Old->_inhibitionRadius = std::stoi(params["inhibitionRadius"]);
				if (params.find("activeRatio") != params.end())
					sfDesc_Old->_activeRatio = std::stof(params["activeRatio"]);
				if (params.find("biasAlpha") != params.end())
					sfDesc_Old->_biasAlpha = std::stof(params["biasAlpha"]);
				if (params.find("initWeightRange") != params.end())
					sfDesc_Old->_initWeightRange = ParameterModifier::parseFloat2(params["initWeightRange"]);
				if (params.find("initBiasRange") != params.end())
					sfDesc_Old->_initBiasRange = ParameterModifier::parseFloat2(params["initBiasRange"]);

				if (layerIndex == 0) {
					sfDesc_Old->_visibleLayerDescs.resize(_inputLayers.size() + 1);
					for (size_t i = 0; i < _inputLayers.size(); i++) {
						sfDesc_Old->_visibleLayerDescs[i]._ignoreMiddle = false;
						sfDesc_Old->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };
						if (_inputLayers[i]._params.find("ff_radius") != _inputLayers[i]._params.end())
							sfDesc_Old->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["ff_radius"]);
						if (_inputLayers[i]._params.find("ff_weightAlpha") != _inputLayers[i]._params.end())
							sfDesc_Old->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["ff_weightAlpha"]);
					}
					// Recurrent
					{
						sfDesc_Old->_visibleLayerDescs.back()._ignoreMiddle = true;
						sfDesc_Old->_visibleLayerDescs.back()._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("r_radius") != params.end())
							sfDesc_Old->_visibleLayerDescs.back()._radius = std::stoi(params["r_radius"]);
						if (params.find("r_weightAlpha") != params.end())
							sfDesc_Old->_visibleLayerDescs.back()._weightAlpha = std::stof(params["r_weightAlpha"]);
					}
				}
				else {
					sfDesc_Old->_visibleLayerDescs.resize(2);
					// Feed forward
					{
						sfDesc_Old->_visibleLayerDescs[0]._ignoreMiddle = false;
						sfDesc_Old->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };
						if (params.find("ff_radius") != params.end())
							sfDesc_Old->_visibleLayerDescs[0]._radius = std::stoi(params["ff_radius"]);
						if (params.find("ff_weightAlpha") != params.end())
							sfDesc_Old->_visibleLayerDescs[0]._weightAlpha = std::stof(params["ff_weightAlpha"]);
					}
					// Recurrent
					{
						sfDesc_Old->_visibleLayerDescs[1]._ignoreMiddle = true;
						sfDesc_Old->_visibleLayerDescs[1]._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("r_radius") != params.end())
							sfDesc_Old->_visibleLayerDescs[1]._radius = std::stoi(params["r_radius"]);
						if (params.find("r_weightAlpha") != params.end())
							sfDesc_Old->_visibleLayerDescs[1]._weightAlpha = std::stof(params["r_weightAlpha"]);
					}
				}
				sfDesc = sfDesc_Old;
				break;
			}
			case _stdp:
			{
				std::shared_ptr<SparseFeaturesSTDP::SparseFeaturesSTDPDesc> sfDescSTDP = std::make_shared<SparseFeaturesSTDP::SparseFeaturesSTDPDesc>();

				sfDescSTDP->_inputType = SparseFeatures::_feedForwardRecurrent;
				sfDescSTDP->_hiddenSize = size;
				sfDescSTDP->_rng = _rng;

				if (params.find("sfs_inhibitionRadius") != params.end())
					sfDescSTDP->_inhibitionRadius = std::stoi(params["sfs_inhibitionRadius"]);
				if (params.find("sfs_initWeightRange") != params.end())
					sfDescSTDP->_initWeightRange = ParameterModifier::parseFloat2(params["sfs_initWeightRange"]);
				if (params.find("sfs_biasAlpha") != params.end())
					sfDescSTDP->_biasAlpha = std::stof(params["sfs_biasAlpha"]);
				if (params.find("sfs_activeRatio") != params.end())
					sfDescSTDP->_activeRatio = std::stof(params["sfs_activeRatio"]);
				if (params.find("sfs_gamma") != params.end())
					sfDescSTDP->_gamma = std::stof(params["sfs_gamma"]);

				if (layerIndex == 0) {
					sfDescSTDP->_visibleLayerDescs.resize(_inputLayers.size() + 1);

					for (size_t i = 0; i < _inputLayers.size(); i++) {
						sfDescSTDP->_visibleLayerDescs[i]._ignoreMiddle = false;
						sfDescSTDP->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };
						if (_inputLayers[i]._params.find("sfs_ff_radius") != _inputLayers[i]._params.end())
							sfDescSTDP->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["sfs_ff_radius"]);
						if (_inputLayers[i]._params.find("sfs_ff_weightAlpha") != _inputLayers[i]._params.end())
							sfDescSTDP->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["sfs_ff_weightAlpha"]);
						if (_inputLayers[i]._params.find("sfs_ff_lambda") != _inputLayers[i]._params.end())
							sfDescSTDP->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfs_ff_lambda"]);
					}
					// Recurrent
					{
						sfDescSTDP->_visibleLayerDescs.back()._ignoreMiddle = true;
						sfDescSTDP->_visibleLayerDescs.back()._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("sfs_r_radius") != params.end())
							sfDescSTDP->_visibleLayerDescs.back()._radius = std::stoi(params["sfs_r_radius"]);
						if (params.find("sfs_r_weightAlpha") != params.end())
							sfDescSTDP->_visibleLayerDescs.back()._weightAlpha = std::stof(params["sfs_r_weightAlpha"]);
						if (params.find("sfs_r_lambda") != params.end())
							sfDescSTDP->_visibleLayerDescs.back()._lambda = std::stof(params["sfs_r_lambda"]);
					}
				}
				else {
					sfDescSTDP->_visibleLayerDescs.resize(2);
					// Feed forward
					{
						sfDescSTDP->_visibleLayerDescs[0]._ignoreMiddle = false;
						sfDescSTDP->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };
						if (params.find("sfs_ff_radius") != params.end())
							sfDescSTDP->_visibleLayerDescs[0]._radius = std::stoi(params["sfs_ff_radius"]);
						if (params.find("sfs_ff_weightAlpha") != params.end())
							sfDescSTDP->_visibleLayerDescs[0]._weightAlpha = std::stof(params["sfs_ff_weightAlpha"]);
						if (params.find("sfs_ff_lambda") != params.end())
							sfDescSTDP->_visibleLayerDescs[0]._lambda = std::stof(params["sfs_ff_lambda"]);
					}
					// Recurrent
					{
						sfDescSTDP->_visibleLayerDescs[1]._ignoreMiddle = true;
						sfDescSTDP->_visibleLayerDescs[1]._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("sfs_r_radius") != params.end())
							sfDescSTDP->_visibleLayerDescs[1]._radius = std::stoi(params["sfs_r_radius"]);
						if (params.find("sfs_r_weightAlpha") != params.end())
							sfDescSTDP->_visibleLayerDescs[1]._weightAlpha = std::stof(params["sfs_r_weightAlpha"]);
						if (params.find("sfs_r_lambda") != params.end())
							sfDescSTDP->_visibleLayerDescs[1]._lambda = std::stof(params["sfs_r_lambda"]);
					}
				}
				sfDesc = sfDescSTDP;
				break;
			}
			case _delay:
			{
				std::shared_ptr<SparseFeaturesDelay::SparseFeaturesDelayDesc> sfDescDelay = std::make_shared<SparseFeaturesDelay::SparseFeaturesDelayDesc>();

				sfDescDelay->_inputType = SparseFeatures::_feedForward;
				sfDescDelay->_hiddenSize = size;
				sfDescDelay->_rng = _rng;

				if (params.find("sfd_inhibitionRadius") != params.end())
					sfDescDelay->_inhibitionRadius = std::stoi(params["sfd_inhibitionRadius"]);
				if (params.find("sfd_initWeightRange") != params.end())
					sfDescDelay->_initWeightRange = ParameterModifier::parseFloat2(params["sfd_initWeightRange"]);
				if (params.find("sfd_biasAlpha") != params.end())
					sfDescDelay->_biasAlpha = std::stof(params["sfd_biasAlpha"]);
				if (params.find("sfd_activeRatio") != params.end())
					sfDescDelay->_activeRatio = std::stof(params["sfd_activeRatio"]);

				if (layerIndex == 0) {
					sfDescDelay->_visibleLayerDescs.resize(_inputLayers.size());
					for (size_t i = 0; i < _inputLayers.size(); i++) {
						sfDescDelay->_visibleLayerDescs[i]._ignoreMiddle = false;
						sfDescDelay->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };
						if (_inputLayers[i]._params.find("sfd_ff_radius") != _inputLayers[i]._params.end())
							sfDescDelay->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["sfd_ff_radius"]);
						if (_inputLayers[i]._params.find("sfd_ff_weightAlpha") != _inputLayers[i]._params.end())
							sfDescDelay->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["sfd_ff_weightAlpha"]);
						if (_inputLayers[i]._params.find("sfd_ff_lambda") != _inputLayers[i]._params.end())
							sfDescDelay->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfd_ff_lambda"]);
						if (_inputLayers[i]._params.find("sfd_ff_gamma") != _inputLayers[i]._params.end())
							sfDescDelay->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfd_ff_gamma"]);
					}
				}
				else {
					sfDescDelay->_visibleLayerDescs.resize(1);
					// Feed forward
					{
						sfDescDelay->_visibleLayerDescs[0]._ignoreMiddle = false;
						sfDescDelay->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };
						if (params.find("sfd_ff_radius") != params.end())
							sfDescDelay->_visibleLayerDescs[0]._radius = std::stoi(params["sfd_ff_radius"]);
						if (params.find("sfd_ff_weightAlpha") != params.end())
							sfDescDelay->_visibleLayerDescs[0]._weightAlpha = std::stof(params["sfd_ff_weightAlpha"]);
						if (params.find("sfd_ff_lambda") != params.end())
							sfDescDelay->_visibleLayerDescs[0]._lambda = std::stof(params["sfd_ff_lambda"]);
						if (params.find("sfd_ff_gamma") != params.end())
							sfDescDelay->_visibleLayerDescs[0]._gamma = std::stof(params["sfd_ff_gamma"]);
					}
				}
				sfDesc = sfDescDelay;
				break;
			}
			case _chunk:
			{
				std::shared_ptr<SparseFeaturesChunk::SparseFeaturesChunkDesc> sfDescChunk = std::make_shared<SparseFeaturesChunk::SparseFeaturesChunkDesc>();

				sfDescChunk->_inputType = SparseFeatures::_feedForwardRecurrent;
				sfDescChunk->_hiddenSize = size;
				sfDescChunk->_rng = _rng;

				if (params.find("sfc_chunkSize") != params.end()) 
					sfDescChunk->_chunkSize = ParameterModifier::parseInt2(params["sfc_chunkSize"]);
				if (params.find("sfc_initWeightRange") != params.end()) 
					sfDescChunk->_initWeightRange = ParameterModifier::parseFloat2(params["sfc_initWeightRange"]);
				if (params.find("sfc_numSamples") != params.end())
					sfDescChunk->_numSamples = std::stoi(params["sfc_numSamples"]);
				if (params.find("sfc_biasAlpha") != params.end())
					sfDescChunk->_biasAlpha = std::stof(params["sfc_biasAlpha"]);
				if (params.find("sfc_gamma") != params.end())
					sfDescChunk->_gamma = std::stof(params["sfc_gamma"]);

				if (layerIndex == 0) {
					sfDescChunk->_visibleLayerDescs.resize(_inputLayers.size() + 1);

					for (size_t i = 0; i < _inputLayers.size(); i++) {
						sfDescChunk->_visibleLayerDescs[i]._ignoreMiddle = false;
						sfDescChunk->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };
						if (_inputLayers[i]._params.find("sfc_ff_radius") != _inputLayers[i]._params.end())
							sfDescChunk->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["sfc_ff_radius"]);
						if (_inputLayers[i]._params.find("sfc_ff_weightAlpha") != _inputLayers[i]._params.end())
							sfDescChunk->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["sfc_ff_weightAlpha"]);
						if (_inputLayers[i]._params.find("sfc_ff_lambda") != _inputLayers[i]._params.end())
							sfDescChunk->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfc_ff_lambda"]);
					}
					// Recurrent
					{
						sfDescChunk->_visibleLayerDescs.back()._ignoreMiddle = true;
						sfDescChunk->_visibleLayerDescs.back()._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("sfc_r_radius") != params.end())
							sfDescChunk->_visibleLayerDescs.back()._radius = std::stoi(params["sfc_r_radius"]);
						if (params.find("sfc_r_weightAlpha") != params.end())
							sfDescChunk->_visibleLayerDescs.back()._weightAlpha = std::stof(params["sfc_r_weightAlpha"]);
						if (params.find("sfc_r_lambda") != params.end())
							sfDescChunk->_visibleLayerDescs.back()._lambda = std::stof(params["sfc_r_lambda"]);
					}
				}
				else {
					sfDescChunk->_visibleLayerDescs.resize(2);
					// Feed forward
					{
						sfDescChunk->_visibleLayerDescs[0]._ignoreMiddle = false;
						sfDescChunk->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };
						if (params.find("sfc_ff_radius") != params.end())
							sfDescChunk->_visibleLayerDescs[0]._radius = std::stoi(params["sfc_ff_radius"]);
						if (params.find("sfc_ff_weightAlpha") != params.end())
							sfDescChunk->_visibleLayerDescs[0]._weightAlpha = std::stof(params["sfc_ff_weightAlpha"]);
						if (params.find("sfc_ff_lambda") != params.end())
							sfDescChunk->_visibleLayerDescs[0]._lambda = std::stof(params["sfc_ff_lambda"]);
					}
					// Recurrent
					{
						sfDescChunk->_visibleLayerDescs[1]._ignoreMiddle = true;
						sfDescChunk->_visibleLayerDescs[1]._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };
						if (params.find("sfc_r_radius") != params.end())
							sfDescChunk->_visibleLayerDescs[1]._radius = std::stoi(params["sfc_r_radius"]);
						if (params.find("sfc_r_weightAlpha") != params.end())
							sfDescChunk->_visibleLayerDescs[1]._weightAlpha = std::stof(params["sfc_r_weightAlpha"]);
						if (params.find("sfc_r_lambda") != params.end())
							sfDescChunk->_visibleLayerDescs[1]._lambda = std::stof(params["sfc_r_lambda"]);
					}
				}
				sfDesc = sfDescChunk;
				break;
			}
			}
			return sfDesc;
		}

	public:

		void initialize(unsigned int seed) {
			_rng.seed(seed);
		}

		ParameterModifier addInputLayer(const int2 &size) {
			InputLayer inputLayer;
			inputLayer._size = size;
			_inputLayers.push_back(inputLayer);
			ParameterModifier pm;
			pm._target = &_inputLayers.back()._params;
			return pm;
		}

		ParameterModifier addActionLayer(const int2 &size, const int2 &tileSize) {
			ActionLayer actionLayer;
			actionLayer._size = size;
			actionLayer._tileSize = tileSize;
			_actionLayers.push_back(actionLayer);
			ParameterModifier pm;
			pm._target = &_actionLayers.back()._params;
			return pm;
		}

		ParameterModifier addHigherLayer(const int2 &size, SparseFeaturesType type) {
			HigherLayer higherLayer;
			higherLayer._size = size;
			higherLayer._type = type;
			_higherLayers.push_back(higherLayer);
			ParameterModifier pm;
			pm._target = &_higherLayers.back()._params;
			return pm;
		}

		std::shared_ptr<class Hierarchy> generateHierarchy() 
		{
			std::unordered_map<std::string, std::string> emptyHierarchy;
			return generateHierarchy(emptyHierarchy);
		}

		std::shared_ptr<class Agent> generateAgent() 
		{
			std::unordered_map<std::string, std::string> emptyAgentHierarchy;
			return generateAgent(emptyAgentHierarchy);
		}

		std::shared_ptr<class Hierarchy> generateHierarchy(
			std::unordered_map<std::string, std::string> &additionalParams) 
		{
			// last checked: 28-nov 2016

			std::shared_ptr<Hierarchy> h = std::make_shared<Hierarchy>();
			h->_rng = _rng;

			for (size_t i = 0; i < _inputLayers.size(); ++i) {
				h->_predictions.push_back(Array2D<float>(_inputLayers[i]._size));
			}

			std::vector<Predictor::PredLayerDesc> pLayerDescs(_higherLayers.size());
			std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(_higherLayers.size());

			float2 initWeightRange = float2{ -0.01f, 0.01f };
			if (additionalParams.find("ad_initWeightRange") != additionalParams.end())
				initWeightRange = ParameterModifier::parseFloat2(additionalParams["ad_initWeightRange"]);

			// Fill out layer descs
			for (size_t l = 0; l < _higherLayers.size(); ++l) {
				if (_higherLayers[l]._params.find("hl_poolSteps") != _higherLayers[l]._params.end())
					hLayerDescs[l]._poolSteps = std::stoi(_higherLayers[l]._params["hl_poolSteps"]);

				hLayerDescs[l]._sfDesc = sfDescFromName(l, _higherLayers[l]._type, _higherLayers[l]._size, SparseFeatures::_feedForwardRecurrent, _higherLayers[l]._params);

				// P layer desc
				if (_higherLayers[l]._params.find("p_alpha") != _higherLayers[l]._params.end())
					pLayerDescs[l]._alpha = std::stof(_higherLayers[l]._params["p_alpha"]);
				if (_higherLayers[l]._params.find("p_radius") != _higherLayers[l]._params.end())
					pLayerDescs[l]._radius = std::stoi(_higherLayers[l]._params["p_radius"]);
			}

			h->_p.createRandom(pLayerDescs, hLayerDescs, initWeightRange, _rng);

			// Create readout layers
			h->_readoutLayers.resize(h->_predictions.size());

			for (size_t i = 0; i < h->_readoutLayers.size(); ++i) {
				std::vector<PredictorLayer::VisibleLayerDesc> vlds(1);

				vlds.front()._size = { _higherLayers.front()._size.x, _higherLayers.front()._size.y };

				if (_inputLayers[i]._params.find("in_p_alpha") != _inputLayers[i]._params.end())
					vlds.front()._alpha = std::stof(_inputLayers[i]._params["in_p_alpha"]);
				if (_inputLayers[i]._params.find("in_p_radius") != _inputLayers[i]._params.end())
					vlds.front()._radius = std::stoi(_inputLayers[i]._params["in_p_radius"]);
				h->_readoutLayers[i].createRandom(h->_predictions[i].getSize(), vlds, nullptr, initWeightRange, _rng);
			}
			return h;
		}
		
		std::shared_ptr<class Agent> Architect::generateAgent(
			std::unordered_map<std::string, std::string> &additionalParams) 
		{
			std::shared_ptr<Agent> a = std::make_shared<Agent>();

			a->_rng = _rng;
			a->_inputImages.resize(_inputLayers.size());

			for (size_t i = 0; i < _inputLayers.size(); i++)
				a->_inputImages[i] = Array2D<float>(_inputLayers[i]._size);

			std::vector<int2> actionSizes(_actionLayers.size());
			std::vector<int2> actionTileSizes(_actionLayers.size());

			for (size_t i = 0; i < _actionLayers.size(); i++) {
				a->_actions.push_back(Array2D<float>(_actionLayers[i]._size));

				actionSizes[i] = { _actionLayers[i]._size.x, _actionLayers[i]._size.y };
				actionTileSizes[i] = { _actionLayers[i]._tileSize.x, _actionLayers[i]._tileSize.y };
			}

			std::vector<std::vector<AgentSwarm::AgentLayerDesc>> aLayerDescs(_higherLayers.size());
			std::vector<Predictor::PredLayerDesc> pLayerDescs(_higherLayers.size());
			std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(_higherLayers.size());

			float2 initWeightRange = { -0.01f, 0.01f };

			if (additionalParams.find("ad_initWeightRange") != additionalParams.end()) {
				float2 range = ParameterModifier::parseFloat2(additionalParams["ad_initWeightRange"]);

				initWeightRange = { range.x, range.y };
			}

			// Fill out layer descs
			for (size_t l = 0; l < _higherLayers.size(); l++) {
				if (_higherLayers[l]._params.find("hl_poolSteps") != _higherLayers[l]._params.end())
					hLayerDescs[l]._poolSteps = std::stoi(_higherLayers[l]._params["hl_poolSteps"]);

				hLayerDescs[l]._sfDesc = sfDescFromName(l, _higherLayers[l]._type, _higherLayers[l]._size, SparseFeatures::_feedForwardRecurrent, _higherLayers[l]._params);

				// P layer desc
				if (_higherLayers[l]._params.find("p_alpha") != _higherLayers[l]._params.end())
					pLayerDescs[l]._alpha = std::stof(_higherLayers[l]._params["p_alpha"]);
				if (_higherLayers[l]._params.find("p_radius") != _higherLayers[l]._params.end())
					pLayerDescs[l]._radius = std::stoi(_higherLayers[l]._params["p_radius"]);

				// A layer desc
				if (l == _higherLayers.size() - 1) {
					aLayerDescs[l].resize(_actionLayers.size());

					for (size_t i = 0; i < _actionLayers.size(); i++) {
						if (_actionLayers[i]._params.find("a_radius") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._radius = std::stoi(_actionLayers[i]._params["a_radius"]);

						if (_actionLayers[i]._params.find("a_qAlpha") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._qAlpha = std::stof(_actionLayers[i]._params["a_qAlpha"]);

						if (_actionLayers[i]._params.find("a_actionAlpha") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._actionAlpha = std::stof(_actionLayers[i]._params["a_actionAlpha"]);

						if (_actionLayers[i]._params.find("a_qGamma") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._qGamma = std::stof(_actionLayers[i]._params["a_qGamma"]);

						if (_actionLayers[i]._params.find("a_qLambda") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._qLambda = std::stof(_actionLayers[i]._params["a_qLambda"]);

						if (_actionLayers[i]._params.find("a_actionLambda") != _actionLayers[i]._params.end())
							aLayerDescs[l][i]._actionLambda = std::stof(_actionLayers[i]._params["a_actionLambda"]);
					}
				}
				else {
					aLayerDescs[l].resize(1);

					if (_higherLayers[l]._params.find("a_radius") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._radius = std::stoi(_higherLayers[l]._params["a_radius"]);

					if (_higherLayers[l]._params.find("a_qAlpha") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._qAlpha = std::stof(_higherLayers[l]._params["a_qAlpha"]);

					if (_higherLayers[l]._params.find("a_actionAlpha") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._actionAlpha = std::stof(_higherLayers[l]._params["a_actionAlpha"]);

					if (_higherLayers[l]._params.find("a_qGamma") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._qGamma = std::stof(_higherLayers[l]._params["a_qGamma"]);

					if (_higherLayers[l]._params.find("a_qLambda") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._qLambda = std::stof(_higherLayers[l]._params["a_qLambda"]);

					if (_higherLayers[l]._params.find("a_actionLambda") != _higherLayers[l]._params.end())
						aLayerDescs[l].front()._actionLambda = std::stof(_higherLayers[l]._params["a_actionLambda"]);
				}
			}

			a->_as.createRandom(actionSizes, actionTileSizes, aLayerDescs, pLayerDescs, hLayerDescs, initWeightRange, _rng);
			return a;
		}
	};
}