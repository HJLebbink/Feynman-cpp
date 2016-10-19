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

#include "Predictor.ipp"
#include "LayerDescs.ipp"

namespace feynman {

	//Default Hierarchy implementation (FeatureHierarchy)
	class Hierarchy {
	private:

		Predictor _ph;
		int _inputWidth, _inputHeight;

		std::mt19937 _rng;

		Image2D<float> _inputImage;

		//Prediction vector
		std::vector<float> _pred;

	public:
		/*!
		\brief Create the Hierarchy
		\param inputWidth is the width of input to the hierarchy.
		\param inputHeight is the height of input to the hierarchy.
		\param layerDescs provide layer descriptors for hierachy.
		\param initMinWeight is the minimum value for weight initialization.
		\param initMaxWeight is the maximum value for weight initialization.
		\param seed a random number generator seed.
		*/
		Hierarchy(
			const int inputWidth, 
			const int inputHeight,
			const std::vector<LayerDescs> &layerDescs,
			const float initMinWeight,
			const float initMaxWeight,
			const int seed) 
		{
			_rng.seed(seed);

			_inputWidth = inputWidth;
			_inputHeight = inputHeight;

			int2 inputSize = { inputWidth, inputHeight };

			std::vector<Predictor::PredLayerDesc> pLayerDescs(layerDescs.size());
			std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(layerDescs.size());

			for (int l = 0; l < layerDescs.size(); l++) {
				hLayerDescs[l]._size = int2{ layerDescs[l]._width, layerDescs[l]._height };
				hLayerDescs[l]._inputDescs = { FeatureHierarchy::InputDesc(inputSize, layerDescs[l]._feedForwardRadius) };
				hLayerDescs[l]._recurrentRadius = layerDescs[l]._recurrentRadius;
				hLayerDescs[l]._inhibitionRadius = layerDescs[l]._inhibitionRadius;
				hLayerDescs[l]._spFeedForwardWeightAlpha = layerDescs[l]._spFeedForwardWeightAlpha;
				hLayerDescs[l]._spRecurrentWeightAlpha = layerDescs[l]._spRecurrentWeightAlpha;
				hLayerDescs[l]._spBiasAlpha = layerDescs[l]._spBiasAlpha;
				hLayerDescs[l]._spActiveRatio = layerDescs[l]._spActiveRatio;

				pLayerDescs[l]._radius = layerDescs[l]._predRadius;
				pLayerDescs[l]._alpha = layerDescs[l]._predAlpha;
				pLayerDescs[l]._beta = layerDescs[l]._predBeta;
			}

			//_ph.createRandom(cs, prog, inputSize, pLayerDescs, hLayerDescs, cl_float2{ initMinWeight, initMaxWeight }, _rng);

			// Create temporary buffers
			//_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputWidth, inputHeight);

			_pred.clear();
			_pred.assign(inputWidth * inputHeight, 0.0f);
		}

		/*!
		\brief Run a single simulation tick
		\param inputs the inputs to the bottom-most layer.
		\param learn optional argument to disable learning.
		*/
		void simStep(
			const std::vector<float> &inputs, 
			const bool learn) 
		{
			assert(inputs.size() == _inputWidth * _inputHeight);

			// Write input
			//std::vector<float> inputsf = inputs;
			//_pCs->getQueue().enqueueWriteImage(_inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 }, 0, 0, inputsf.data());
			copy(inputs, _inputImage); //TODO remove the unnecessary copy

			_ph.simStep(_inputImage, _inputImage, _rng, learn);

			// Get prediction
			//_pCs->getQueue().enqueueReadImage(_ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 }, 0, 0, _pred.data());
			copy(_ph.getPrediction(), _pred);
		}

		//Get the current prediction vector
		const std::vector<float> &getPrediction() const {
			return _pred;
		}
	};
}