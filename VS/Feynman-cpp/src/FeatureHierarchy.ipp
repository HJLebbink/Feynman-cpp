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
#include "SparseFeatures.ipp"

namespace feynman {

	//Hierarchy of sparse features
	class FeatureHierarchy {
	public:

		//Descriptor of a layer input
		struct InputDesc {

			//Size of layer
			int2 _size;

			// Radii for feed forward and inhibitory connections
			int _radius;

			//Initialize defaults
			InputDesc() : _size({ 8, 8 }), _radius(0)
			{}

			//brief Initialize from values
			InputDesc(int2 size, int radius)
				: _size(size), _radius(radius)
			{}
		};

		//Descriptor of a layer in the feature hierarchy
		struct LayerDesc {
			
			//Size of layer
			int2 _size;

			//Input descriptors
			std::vector<InputDesc> _inputDescs;

			//Radius for recurrent connections
			int _recurrentRadius;

			//Radius for inhibitory connections
			int _inhibitionRadius;

			//Sparse predictor parameters
			float _spFeedForwardWeightAlpha;
			float _spRecurrentWeightAlpha;
			float _spBiasAlpha;
			float _spActiveRatio;

			//Initialize defaults
			LayerDesc()
				: _size({ 8, 8 }),
				_inputDescs({ InputDesc({ 16, 16 }, 6) }), _recurrentRadius(6), _inhibitionRadius(5),
				_spFeedForwardWeightAlpha(0.25f), _spRecurrentWeightAlpha(0.25f), _spBiasAlpha(0.01f),
				_spActiveRatio(0.02f)
			{}
		};

		//Layer
		struct Layer {
			
			//Sparse predictor
			SparseFeatures _sparseFeatures;

		};

	private: 

		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

	public:
		//Initialize defaults
		FeatureHierarchy()
		{}

		/**
		Create a sparse feature hierarchy with random initialization.
		Requires initialization information.
		\param inputDescs are the descriptors of the input layers.
		\param layerDescs are descriptors for feature hierarchy layers.
		\param initWeightRange are the minimum and maximum range values for weight initialization.
		\param rng a random number generator.
		*/
		void createRandom(
			const std::vector<InputDesc> &inputDescs, 
			const std::vector<LayerDesc> &layerDescs,
			const float2 initWeightRange,
			std::mt19937 &rng)
		{
			_layerDescs = layerDescs;
			_layerDescs.front()._inputDescs = inputDescs;
			_layers.resize(_layerDescs.size());
			int2 prevLayerSize = inputDescs.front()._size;

			for (size_t layer = 0; layer < _layers.size(); layer++) {
				if (layer == 0) {
					std::vector<SparseFeatures::VisibleLayerDesc> spDescs(inputDescs.size());

					for (size_t i = 0; i < inputDescs.size(); i++) {
						// Feed forward
						spDescs[i]._size = inputDescs[i]._size;
						spDescs[i]._radius = inputDescs[i]._radius;
						spDescs[i]._ignoreMiddle = false;
						spDescs[i]._weightAlpha = _layerDescs[layer]._spFeedForwardWeightAlpha;
					}

					// Recurrent
					if (_layerDescs[layer]._recurrentRadius != 0) {
						SparseFeatures::VisibleLayerDesc recDesc;

						recDesc._size = _layerDescs[layer]._size;
						recDesc._radius = _layerDescs[layer]._recurrentRadius;
						recDesc._ignoreMiddle = true;
						recDesc._weightAlpha = _layerDescs[layer]._spRecurrentWeightAlpha;

						spDescs.push_back(recDesc);
					}

					_layers[layer]._sparseFeatures.createRandom(spDescs, _layerDescs[layer]._size, _layerDescs[layer]._inhibitionRadius, initWeightRange, rng);
				}
				else {
					std::vector<SparseFeatures::VisibleLayerDesc> spDescs(_layerDescs[layer]._recurrentRadius != 0 ? 2 : 1);

					// Feed forward
					spDescs[0]._size = prevLayerSize;
					spDescs[0]._radius = _layerDescs[layer]._inputDescs.front()._radius;
					spDescs[0]._ignoreMiddle = false;
					spDescs[0]._weightAlpha = _layerDescs[layer]._spFeedForwardWeightAlpha;

					// Recurrent
					if (_layerDescs[layer]._recurrentRadius != 0) {
						spDescs[1]._size = _layerDescs[layer]._size;
						spDescs[1]._radius = _layerDescs[layer]._recurrentRadius;
						spDescs[1]._ignoreMiddle = true;
						spDescs[1]._weightAlpha = _layerDescs[layer]._spRecurrentWeightAlpha;
					}

					_layers[layer]._sparseFeatures.createRandom(spDescs, _layerDescs[layer]._size, _layerDescs[layer]._inhibitionRadius, initWeightRange, rng);
				}

				// Next layer
				prevLayerSize = _layerDescs[layer]._size;
			}
		}

		/*!
		\brief Simulation step of hierarchy
		Runs one timestep of simulation.
		\param inputs the inputs to the bottom-most layer.
		\param rng a random number generator.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const std::vector<Image2D> &inputs,
			std::mt19937 &rng,
			const bool learn = true)
		{
			std::vector<Image2D> inputsUse = inputs;

			if (_layerDescs.front()._recurrentRadius != 0) {
				inputsUse.push_back(_layers.front()._sparseFeatures.getHiddenStates()[_back]);
			}
			// Activate
			for (size_t layer = 0; layer < _layers.size(); layer++) {

				std::vector<Image2D> visibleStates;
				if (layer == 0) {
					visibleStates = inputsUse;
				}
				else if (_layerDescs[layer]._recurrentRadius == 0) {
					visibleStates = std::vector<Image2D>{
						_layers[layer - 1]._sparseFeatures.getHiddenStates()[_front]
					};
				} else {
					visibleStates = std::vector<Image2D>{
						_layers[layer - 1]._sparseFeatures.getHiddenStates()[_front],
						_layers[layer]._sparseFeatures.getHiddenStates()[_back]
					};
				}
				_layers[layer]._sparseFeatures.activate(
					visibleStates, 
					_layerDescs[layer]._spActiveRatio, 
					rng);
			}
			// Learn
			if (learn) {
				for (size_t layer = 0; layer < _layers.size(); layer++) {
					_layers[layer]._sparseFeatures.learn(
						_layerDescs[layer]._spBiasAlpha, 
						_layerDescs[layer]._spActiveRatio);
				}
			}
			// Step end
			for (size_t layer = 0; layer < _layers.size(); layer++) {
				_layers[layer]._sparseFeatures.stepEnd();
			}
		}

		// Get number of layers
		size_t getNumLayers() const {
			return _layers.size();
		}

		/*!
		\brief Get access to a layer
		\param[in] index Layer index.
		*/
		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		/*!
		\brief Get access to a layer desc
		\param[in] index Layer index.
		*/
		const LayerDesc &getLayerDesc(int index) const {
			return _layerDescs[index];
		}

		//Clear the working memory
		void clearMemory() {
			for (size_t layer = 0; layer < _layers.size(); layer++) {
				_layers[layer]._sparseFeatures.clearMemory();
			}
		}
	};
}