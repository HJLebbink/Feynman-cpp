// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>		// for cerr and cout
#include <tuple>
#include <vector>
#include <random>
#include <algorithm> // for std::min
#include <ctime>
#include <memory>

#include "timing.h"
#include "Helpers.ipp"

namespace feynman {

	//Sparse Features
	//Base class for encoders(sparse features)
	class SparseFeatures {
	public:

		enum InputType {
			_feedForward, _feedForwardRecurrent
		};

		SparseFeaturesType _type;

	public:

		/*!
		\brief Sparse Features Descriptor
		Base class for encoder descriptors (sparse features descriptors)
		*/
		class SparseFeaturesDesc {
		public:
			std::string _name;

			InputType _inputType;

			virtual size_t getNumVisibleLayers() const = 0;
			virtual int2 getVisibleLayerSize(int vli) const = 0;
			virtual int2 getHiddenSize() const = 0;
			virtual std::shared_ptr<SparseFeatures> sparseFeaturesFactory() = 0;

			//Initialize defaults
			SparseFeaturesDesc()
				: _name("Unassigned"), _inputType(_feedForward)
			{}

			virtual ~SparseFeaturesDesc() {}
		};

		virtual ~SparseFeatures() {}

		/*!
		\brief Activate predictor
		\param visibleStates the input layer states.
		\param todo
		\param rng a random number generator.
		*/
		virtual void activate(
			const std::vector<Array2D<float2>> &visibleStates,
			const Array2D<float2> &predictionsPrev,
			std::mt19937 &rng) = 0;

		//End a simulation step
		virtual void stepEnd() = 0;

		//Learning
		virtual void learn(
			std::mt19937 &rng) = 0;

		//Inhibition
		virtual void inhibit(
			const Array2D<float2> &activations,
			Array2D<float2> &states,
			std::mt19937 &rng) = 0;

		//Get hidden size
		virtual int2 getHiddenSize() const = 0;

		//Get hidden states
		virtual const DoubleBuffer2D<float2> &getHiddenStates() const = 0;

		// Get context
		virtual const Array2D<float2> &getHiddenContext() const {
			/// last checked : 28-nov 2016
			return getHiddenStates()[_back];
		}

		//Clear the working memory
		virtual void clearMemory() = 0;
	};
}