// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once
#include <memory>
#include "Helpers.ipp"
#include "SparseFeatures.ipp"
#include "FixedPoint.ipp"


namespace feynman {

	/*!
	\brief Chunk encoder (sparse features)
	Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
	*/
	class SparseFeaturesChunk : public SparseFeatures {
	public:

		//Visible layer desc
		struct VisibleLayerDesc {
			//Size of layer
			int2 _size;

			//Radius onto input
			int _radius;

			//Whether or not the middle (center) input should be ignored (self in recurrent schemes)
			bool _ignoreMiddle;

			//Learning rate
			float _weightAlpha;

			//Short trace rate
			float _lambda;

			//Initialize defaults
			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(8), _ignoreMiddle(false),
				_weightAlpha(0.1f), _lambda(0.0f)
			{}
		};

		//Visible layer
		struct VisibleLayer {

			//Possibly manipulated input
			DoubleBuffer2D2f _derivedInput;

			// Samples (time sliced derived inputs)
			DoubleBuffer3D _samples;

			//Reconstruction errors
			Image3D _recons;

			//Weights
			DoubleBuffer3D _weights; // Encoding weights (creates spatio-temporal sparse code)

			//Transformations
			float2 _hiddenToVisible;
			float2 _visibleToHidden;

			int2 _reverseRadii;
		};

		//Sparse Features Chunk Descriptor
		class SparseFeaturesChunkDesc : public SparseFeatures::SparseFeaturesDesc {
		public:
			std::vector<VisibleLayerDesc> _visibleLayerDescs;
			int2 _hiddenSize;
			int2 _chunkSize;
			int _numSamples;
			float _biasAlpha;
			float _gamma;
			float2 _initWeightRange;
			std::mt19937 _rng;

			//Defaults
			SparseFeaturesChunkDesc()
				: _hiddenSize({ 16, 16 }),
				_chunkSize({ 8, 8 }),
				_numSamples(1),
				_biasAlpha(0.0f),
				_gamma(1.6f),
				_initWeightRange({ 0.0f, 1.0f }),
				_rng()
			{
				_name = "chunk";
			}

			size_t getNumVisibleLayers() const override {
				return _visibleLayerDescs.size();
			}

			int2 getVisibleLayerSize(int vli) const override {
				return _visibleLayerDescs[vli]._size;
			}

			int2 getHiddenSize() const override {
				return _hiddenSize;
			}

			//Factory
			std::shared_ptr<SparseFeatures> sparseFeaturesFactory() override {
				return std::make_shared<SparseFeaturesChunk>(_visibleLayerDescs, _hiddenSize, _chunkSize, _numSamples, _biasAlpha, _gamma, _initWeightRange, _rng);
			}
		};

	private:

		//Activations, states, biases
		DoubleBuffer2D2f _hiddenStates;
		DoubleBuffer2D2f _hiddenActivations;
		DoubleBuffer2D _hiddenBiases;
		Array2Di2 _chunkWinners;

		int2 _hiddenSize;

		//Ratio between number of hidden states and number of chunks
		float2 _chunkToHidden;

		//Size of chunks
		int2 _chunkSize;

		//Hidden summation temporary buffer
		DoubleBuffer2D _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

	public:
		int _numSamples;
		float _biasAlpha;
		float _gamma;

		//Default constructor
		SparseFeaturesChunk() {};

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information.
		\param visibleLayerDescs descriptors for each input layer.
		\param hiddenSize hidden layer (SDR) size (2D).
		\param rng a random number generator.
		*/
		SparseFeaturesChunk(
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			const int2 hiddenSize,
			const int2 chunkSize,
			const int numSamples,
			const float biasAlpha,
			const float gamma,
			const float2 initWeightRange,
			std::mt19937 &rng
		) :
			_biasAlpha(biasAlpha), 
			_gamma(gamma)
		{
			// last checked: 28-nov 2016

			_type = SparseFeaturesType::_chunk;
			_visibleLayerDescs = visibleLayerDescs;
			_hiddenSize = hiddenSize;
			_chunkSize = chunkSize;
			_numSamples = numSamples;

			_visibleLayers.resize(_visibleLayerDescs.size());

			const int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / _chunkSize.x));
			const int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / _chunkSize.y));
			//std::cout << "INFO: SparseFeaturesChunk:constructor: chunksInX=" << chunksInX << "; chunksInY=" << chunksInY << std::endl;

			_chunkToHidden = float2{ 
				static_cast<float>(_hiddenSize.x) / chunksInX,
				static_cast<float>(_hiddenSize.y) / chunksInY
			};

			// Create layers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				vl._hiddenToVisible = float2{
					static_cast<float>(vld._size.x) / _hiddenSize.x,
					static_cast<float>(vld._size.y) / _hiddenSize.y
				};
				vl._visibleToHidden = float2{
					static_cast<float>(_hiddenSize.x) / vld._size.x,
					static_cast<float>(_hiddenSize.y) / vld._size.y
				};
				vl._reverseRadii = int2{
					static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
					static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
				};
				{
					const int weightDiam = vld._radius * 2 + 1;
					const int numWeights = weightDiam * weightDiam * _numSamples;
					const int3 weightsSize = int3{ _hiddenSize.x, _hiddenSize.y, numWeights };
					vl._weights = createDoubleBuffer3D(weightsSize);

					//std::cout << "INFO: SparseFeaturesChunk:constructor: initWeightRange=" << initWeightRange.x << "," << initWeightRange.y << std::endl;
					randomUniform3D(vl._weights[_back], weightsSize, initWeightRange, rng);
					//plots::plotImage(vl._weights[_back], 6, "SFChunk:constructor:weights" + std::to_string(vli));
				}
				vl._derivedInput = createDoubleBuffer2D2f(vld._size);
				clear(vl._derivedInput[_back]);

				vl._samples = createDoubleBuffer3D({ vld._size.x, vld._size.y, numSamples });
				clear(vl._samples[_back]);

				//vl._recons = Image3D({vld._size.x, vld._size.y, numSamples});
				//clear(vl._recons);
			}

			// Hidden state data
			_hiddenStates = createDoubleBuffer2D2f(_hiddenSize);
			_hiddenActivations = createDoubleBuffer2D2f(_hiddenSize);
			_hiddenBiases = createDoubleBuffer2D(_hiddenSize);
			_chunkWinners = Array2Di2(int2{ chunksInX, chunksInY });
			_hiddenSummationTemp = createDoubleBuffer2D(_hiddenSize);

			clear(_hiddenStates[_back]);
			clear(_hiddenActivations[_back]);

			if (true) {
				randomUniform2D(_hiddenBiases[_back], _hiddenSize, initWeightRange, rng);
			} else {
				clear(_hiddenBiases[_back]);
			}
		}


		/*!
		\brief Activate predictor
		\param lambda decay of hidden unit traces.
		\param activeRatio % active units.
		\param rng a random number generator.
		*/
		void activate(
			const std::vector<Array2D2f> &visibleStates,
			const Array2D2f &predictionsPrev,
			std::mt19937 &rng) override
		{
			// last checked: 28-nov 2016

			// Start by clearing stimulus summation buffer to biases
			clear(_hiddenSummationTemp[_back]);
			//copy(_hiddenBiases[_back], _hiddenSummationTemp[_back]);

			// Find up stimulus
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				//plots::plotImage(visibleStates[vli], 6, "SFChunk:activate:visibleStates" + std::to_string(vli));

				sfcDeriveInputs(
					visibleStates[vli],			// in
					vl._derivedInput[_back],	// unused
					vl._derivedInput[_front],	// out
					vld._lambda,
					vld._size
				);

				//plots::plotImage(vl._derivedInput[_front], 6, "SFChunk:activate:derivedInput" + std::to_string(vli));

				// Add sample
				sfcAddSample(
					vl._derivedInput[_front],		// in
					vl._samples[_back],				// in
					vl._samples[_front],			// out
					_numSamples,
					vld._size
				);

				//plots::plotImage(vl._samples[_front], 6, "SFChunk:activate:samples" + std::to_string(vli));
				//plots::plotImage(vl._weights[_back], 6, "SFChunk:activate:weights" + std::to_string(vli));

				sfcStimulus(
					vl._samples[_front],			// in
					_hiddenSummationTemp[_back],	// in
					_hiddenSummationTemp[_front],	// out
					vl._weights[_back],				// in
					vld._size,
					vl._hiddenToVisible,
					_chunkSize,
					_chunkToHidden,
					vld._radius,
					_numSamples,
					vld._ignoreMiddle,
					_hiddenSize
				);

				//plots::plotImage(_hiddenSummationTemp[_front], 8, "SFChunk:activate:hiddenSummationTemp" + std::to_string(vli));

				// Swap buffers
				std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
			}

			// Activate
			sfcActivate(
				_hiddenSummationTemp[_back],	// in
				_hiddenStates[_back],			// unused
				_hiddenActivations[_front],		// out
				_hiddenSize
			);
			//plots::plotImage(_hiddenSummationTemp[_back], 6, "SFChunk:activate:hiddenSummationTemp");
			//plots::plotImage(_hiddenActivations[_front], 6, "SFChunk:activate:hiddenActivations");


			// Inhibit
			const int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / _chunkSize.x));
			const int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / _chunkSize.y));

			sfcInhibit(
				_hiddenActivations[_front],		// in
				_hiddenStates[_front],			// out
				_chunkWinners,
				_hiddenSize,
				_chunkSize,
				int2{ chunksInX , chunksInY }
			);
			plots::plotImage(_hiddenActivations[_front], 6, "SFChunk:activate:hiddenActivations");
			plots::plotImage(_hiddenStates[_front], 6, "SFChunk:activate:hiddenStates");
		}
		
		//End a simulation step
		void stepEnd() override
		{
			// last checked: 28-nov 2016

			std::swap(_hiddenStates[_front], _hiddenStates[_back]);
			std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);

			// Swap buffers
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				//VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
				std::swap(vl._samples[_front], vl._samples[_back]);
			}
		}

		/*!
		\brief Learning
		\param biasAlpha learning rate of bias.
		\param activeRatio % active units.
		\param gamma synaptic trace decay.
		*/
		void learn(std::mt19937 &rng) override 
		{
			// last checked: 28-nov

			// Learn weights
			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				// Reconstruct
				/*{
				int argIndex = 0;

				_reconstructKernel.setArg(argIndex++, _hiddenStates[_front]);
				_reconstructKernel.setArg(argIndex++, _hiddenActivations[_front]);
				_reconstructKernel.setArg(argIndex++, vl._recons);
				_reconstructKernel.setArg(argIndex++, vl._weights[_back]);
				_reconstructKernel.setArg(argIndex++, vld._size);
				_reconstructKernel.setArg(argIndex++, _hiddenSize);
				_reconstructKernel.setArg(argIndex++, vl._visibleToHidden);
				_reconstructKernel.setArg(argIndex++, vl._hiddenToVisible);
				_reconstructKernel.setArg(argIndex++, _chunkSize);
				_reconstructKernel.setArg(argIndex++, _chunkToHidden);
				_reconstructKernel.setArg(argIndex++, vld._radius);
				_reconstructKernel.setArg(argIndex++, vl._reverseRadii);
				_reconstructKernel.setArg(argIndex++, _numSamples);

				cs.getQueue().enqueueNDRangeKernel(_reconstructKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
				}*/

				std::cout << "INFO: SFChunk:learn: vld._weightAlpha=" << vld._weightAlpha << "; _gamma="<< _gamma <<std::endl;

				// Weight update
				sfcLearnWeights(
					_hiddenStates[_front],	// in
					_chunkWinners,			// in
					vl._samples[_front],	// in
					vl._weights[_back],		// in
					vl._weights[_front],	// out
					_hiddenSize,
					vld._size,
					vl._hiddenToVisible,
					_chunkSize,
					_chunkToHidden,
					vld._radius,
					vld._weightAlpha,
					_numSamples,
					_gamma,
					_hiddenSize);

				std::swap(vl._weights[_front], vl._weights[_back]);
			}
		}

		/*!
		\brief Inhibit
		Inhibits given activations using this encoder's inhibitory pattern
		*/
		void inhibit(
			const Array2D2f &activations,
			Array2D2f &states,
			std::mt19937 &rng) override 
		{
			// last checked: 25-nov 2016
			const int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / _chunkSize.x));
			const int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / _chunkSize.y));

			sfcInhibitOther(
				activations,	// in
				states,			// out
				_hiddenSize,
				_chunkSize,
				int2{ chunksInX, chunksInY }
			);
		}

		/*!
		\brief Get number of visible layers
		*/
		size_t getNumVisibleLayers() const {
			return _visibleLayers.size();
		}

		/*!
		\brief Get access to visible layer
		*/
		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		/*!
		\brief Get access to visible layer
		*/
		const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		/*!
		\brief Get hidden size
		*/
		int2 getHiddenSize() const override {
			return _hiddenSize;
		}

		/*!
		\brief Get hidden states
		*/
		const DoubleBuffer2D2f &getHiddenStates() const override {
			// last checked: 25-nov
			return _hiddenStates;
		}

		/*!
		\brief Get context
		*/
		const Array2D2f &getHiddenContext() const override {
			// last checked: 25-nov
			return _hiddenActivations[_back];
		}

		/*!
		\brief Clear the working memory
		*/
		void clearMemory() override 
		{
			// last checked: 25-nov
			clear(_hiddenStates[_back]);
			clear(_hiddenActivations[_back]);

			for (size_t vli = 0; vli < _visibleLayers.size(); ++vli) {
				VisibleLayer &vl = _visibleLayers[vli];
				clear(vl._derivedInput[_back]);
				clear(vl._samples[_back]);
			}
		}

		// approx memory usage in bytes;
		size_t getMemoryUsage(bool plot) const {
			size_t nBytes = 0;
			size_t bytes;
			/*
			bytes = _hiddenActivations[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenActivations:   " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenStates[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenStates:        " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenBiases[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenBiases:        " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _hiddenSummationTemp[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_hiddenSummationTemp: " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			for (size_t layer = 0; layer < _visibleLayers.size(); ++layer)
			{
			bytes = _visibleLayers[layer]._derivedInput[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_visibleLayers[" << layer << "]:_derivedInput: " << bytes << " bytes" << std::endl;
			nBytes += bytes;

			bytes = _visibleLayers[layer]._weights[0]._data_float.size() * sizeof(float) * 2;
			if (plot) std::cout << "SparseFeatures:_visibleLayers[" << layer << "]:_weights:      " << bytes << " bytes" << std::endl;
			nBytes += bytes;
			}
			*/
			return nBytes;
		}

		static void speedTest(size_t nExperiments = 1) {
#			ifdef _DEBUG
			nExperiments = 1;
#			endif
			speedTest_spLearnWeights(nExperiments);
			speedTest_spStimulus(nExperiments);
		}
		static void speedTest_spLearnWeights(const size_t nExperiments = 1) {
			printf("Running SparseFeatures::speedTest_spLearnWeights\n");
			/*
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

			const int RADIUS = 6;
			const float weightAlpha = 0.002;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Image2D derivedInput = Image2D(visibleSize);
			Image2D hiddenStates = Image2D(hiddenSize);
			Image3D weightsBack = Image3D(weightsSize);
			Image3D weightsFront0 = Image3D(weightsSize);
			Image3D weightsFront1 = Image3D(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(derivedInput, visibleSize, initRange, generator);
			randomUniform2D(hiddenStates, hiddenSize, initRange, generator);
			randomUniform3D(weightsBack, weightsSize, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spLearnWeights_v0<RADIUS>(
					hiddenStates,		// in
					derivedInput,		// in
					weightsBack,		// in
					weightsFront0,		// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,		// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[spLearnWeights_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spLearnWeights_v1<RADIUS>(
					hiddenStates,		// in
					derivedInput,		// in
					weightsBack,		// in
					weightsFront1,		// out
					visibleSize,
					hiddenToVisible,
					//activeRatio,		// unused
					weightAlpha);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[spLearnWeights_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < weightsFront1._size.x; ++x) {
				for (int y = 0; y < weightsFront1._size.y; ++y) {
					for (int z = 0; z < weightsFront1._size.z; ++z) {
						const float f0 = read_3D(weightsFront0, x, y, z);
						const float f1 = read_3D(weightsFront1, x, y, z);
						const float diff = f0 - f1;
						if (std::abs(diff) > 0.000001) printf("WARNING: SparseFeatures::speedTest_spLearnWeights: coord=(%i,%i): f0=%30.28f; f1=%30.28f; diff=%30.28f\n", x, y, f0, f1, diff);
					}
				}
			}
			*/

		}
		static void speedTest_spStimulus(const size_t nExperiments = 1) {
			printf("Running SparseFeatures::speedTest_spStimulus\n");
			/*
			std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));

			const int RADIUS = 6;
			const int ignoreMiddle = false;
			const int2 visibleSize = { 64, 64 };
			const int2 hiddenSize = { 64, 64 };

			const int weightDiam = RADIUS * 2 + 1;
			const int numWeights = weightDiam * weightDiam;
			const int3 weightsSize = { hiddenSize.x, hiddenSize.y, numWeights };

			Image2D visibleStates = Image2D(visibleSize);
			Image2D hiddenSummationTempBack = Image2D(hiddenSize);
			Image2D hiddenSummationTempFront0 = Image2D(hiddenSize);
			Image2D hiddenSummationTempFront1 = Image2D(hiddenSize);
			Image3D weights = Image3D(weightsSize);
			const float2 hiddenToVisible = float2{
				static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
				static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y)
			};

			//----------------------------------------------------------------------------------
			const float2 initRange = { 0.0f, 1.0f };
			randomUniform2D(visibleStates, visibleSize, initRange, generator);
			randomUniform2D(hiddenSummationTempBack, hiddenSize, initRange, generator);
			randomUniform3D(weights, weightsSize, initRange, generator);

			//----------------------------------------------------------------------------------
			double min0 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spStimulus_v0<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront0,	// out
					weights,					// in
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);

				const double dt = ::tools::get_elapsed_mcycles();
				min0 = std::min(min0, dt);
			}
			printf("[spStimulus_v0]: %2.5f Mcycles\n", min0);

			//----------------------------------------------------------------------------------
			double min1 = std::numeric_limits<double>::max();
			for (size_t i = 0; i < nExperiments; ++i) {
				::tools::reset_and_start_timer();

				spStimulus_v1<RADIUS>(
					visibleStates,				// in
					hiddenSummationTempBack,	// in
					hiddenSummationTempFront1,	// out
					weights,					// in
					visibleSize,
					hiddenToVisible,
					ignoreMiddle);

				const double dt = ::tools::get_elapsed_mcycles();
				min1 = std::min(min1, dt);
			}
			printf("[spStimulus_v1]: %2.5f Mcycles\n", min1);
			printf("\t\t\t\t\t(%.2fx speedup from reference)\n", min0 / min1);

			for (int x = 0; x < hiddenSummationTempFront1._size.x; ++x) {
				for (int y = 0; y < hiddenSummationTempFront1._size.y; ++y) {
					const double f0 = read_2D(hiddenSummationTempFront0, x, y);
					const double f1 = read_2D(hiddenSummationTempFront1, x, y);
					const double diff = f0 - f1;
					if (std::abs(diff) > 0.00001) {
						// the floating point model induces some rounding differences between the methods
						printf("WARNING: SparseFeatures::speedTest_spStimulus: coord=(%i,%i): f0=%30.28f; f1=%30.28f; diff=%30.28f\n", x, y, f0, f1, diff);
					}
				}
			}
			*/
		}

	private:

		static void sfcAddSample(
			const Array2D2f &visibleStates,
			const Image3D &samplesBack, 
			Image3D &samplesFront,
			const int numSamples,
			const int2 range)
		{
			// last checked: 25-nov 2016
			for (int position_x = 0; position_x < range.x; ++position_x) {
				for (int position_y = 0; position_y < range.y; ++position_y) {
					const float visibleState = read_2D(visibleStates, position_x, position_y).x;
					for (int s = 1; s < numSamples; ++s) {
						const float samplePrev = read_3D(samplesBack, position_x, position_y, s - 1);
						write_3D(samplesFront, position_x, position_y, s, samplePrev);
					}
					write_3D(samplesFront, position_x, position_y, 0, visibleState);
				}
			}
			//plots::plotImage(samplesFront, 6, "SparseFeaturesChunk:sfcAddSample:samplesFront");
		}

		template <bool CORNER, int RADIUS>
		static void spStimulus_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image3D &stimulus,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const int numSamples,
			const bool ignoreMiddle)
		{
			// last checked: 23-nov
#			ifdef USE_FIXED_POINT
			spStimulus_fixedp_kernel<CORNER, RADIUS>(
				hiddenPosition_x,
				hiddenPosition_y,
				stimulus,
				hiddenSummationTempBack,
				hiddenSummationTempFront,
				weights,
				visibleSize,
				hiddenToVisible,
				chunkSize,
				chunksToHidden,
				numSamples,
				ignoreMiddle);
#			else
			sfcStimulus_floatp_kernel<CORNER, RADIUS>(
				hiddenPosition_x,
				hiddenPosition_y,
				stimulus,
				hiddenSummationTempBack,
				hiddenSummationTempFront,
				weights,
				visibleSize,
				hiddenToVisible,
				chunkSize,
				chunksToHidden,
				numSamples,
				ignoreMiddle);
#			endif
		}

		template <bool CORNER, int RADIUS>
		static void sfcStimulus_floatp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image3D &samples,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const int numSamples,
			const bool ignoreMiddle)
		{
			// last checked: 23-nov
			throw 1;
			if (true) {
				int chunkPosition_x = hiddenPosition_x / chunkSize.x;
				int chunkPosition_y = hiddenPosition_y / chunkSize.y;

				const int chunkCenter_x = getChunkCenter(chunkPosition_x, chunksToHidden.x);
				const int chunkCenter_y = getChunkCenter(chunkPosition_y, chunksToHidden.y);

				int visiblePositionCenter_x = project(chunkCenter_x, hiddenToVisible.x);
				int visiblePositionCenter_y = project(chunkCenter_y, hiddenToVisible.y);

				float subSum = 0.0f;
				float count = 0.0f;

				int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
				int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;

				for (int s = 0; s < numSamples; s++) {
					for (int dx = -radius; dx <= radius; dx++) {
						for (int dy = -radius; dy <= radius; dy++) {
							int visiblePosition_x = visiblePositionCenter_x + dx;
							int visiblePosition_y = visiblePositionCenter_y + dy;

							if (ignoreMiddle && dx == 0 && dy == 0)
								continue;

							if (inBounds(visiblePosition_x, visibleSize_x) && inBounds(visiblePosition_y, visibleSize_y)) {
								int offset_x = visiblePosition_x - fieldLowerBound_x;
								int offset_x = visiblePosition_y - fieldLowerBound_y;

								int wi = s + numSamples * (offset.y + offset.x * (RADIUS * 2 + 1));
								float weight = read_3D(weights, hiddenPosition.x, hiddenPosition.y, wi);
								float sample = read_3D(samples, visiblePosition.x, visiblePosition.y, s);
								float delta = sample - weight;
								subSum += -delta * delta;
								count += 1.0f;
							}
						}
					}
				}

				float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;
				const float newValue = sum + subSum / fmax(0.0001f, count);
				write_imagef(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newValue);
			}
			else {
				/*
				const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
				const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
				const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
				const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
				const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

				const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
				const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
				const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
				const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
				const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

				float subSum = 0.0f;
				float stateSum = 0.0f;

#			pragma ivdep
				for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
					const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
					for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
						const int offset_y = visiblePosition_y - fieldLowerBound_y;

						const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
						const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
						const float visibleState = read_2D(samples, visiblePosition_x, visiblePosition_y);

						subSum += visibleState * weight;
						stateSum += visibleState;
					}
				}

				if (ignoreMiddle) { // substract the visible state that corresponds to dx=dy=0

					const int visiblePosition_x = visiblePositionCenter_x;
					const int visiblePosition_y = visiblePositionCenter_y;

					const int offset_x = visiblePositionCenter_x - fieldLowerBound_x;
					const int offset_y = visiblePositionCenter_y - fieldLowerBound_y;

					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
					const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
					const float visibleState = read_2D(samples, visiblePosition_x, visiblePosition_y);

					subSum -= visibleState * weight;
					stateSum -= visibleState;
				}

				const float stimulusAddition = subSum / std::max(0.0001f, stateSum);
				//std::cout << "SparseFeatures::spStimulus_float_kernel: floatp=" << stimulusAddition << std::endl;
				const float sum = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
				float sumTemp = sum + stimulusAddition;
				write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, sumTemp);
				*/
			}
		}

		template <bool CORNER, int RADIUS>
		static void spStimulus_fixedp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image2D &visibleStates,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			throw 1;

			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			FixedP3 subSum = 0;
			FixedP3 stateSum = 0; // FixPoint2 should also provide sufficient room

#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;

					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
					const FixedP weight = read_3D_fixp(weights, hiddenPosition_x, hiddenPosition_y, wi);
					const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);

					subSum += static_cast<FixedP3>(visibleState) * static_cast<FixedP3>(weight);
					stateSum += static_cast<FixedP3>(visibleState);
				}
			}

			if (ignoreMiddle) { // substract the visible state that corresponds to dx=dy=0

				const int visiblePosition_x = visiblePositionCenter_x;
				const int visiblePosition_y = visiblePositionCenter_y;

				const int offset_x = visiblePosition_x - fieldLowerBound_x;
				const int offset_y = visiblePosition_y - fieldLowerBound_y;

				const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));
				const FixedP weight = read_3D_fixp(weights, hiddenPosition_x, hiddenPosition_y, wi);
				const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);

				subSum -= static_cast<FixedP3>(visibleState) * static_cast<FixedP3>(weight);
				stateSum -= static_cast<FixedP3>(visibleState);
			}

			const FixedP sumFixP = read_2D_fixp(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
			FixedP newState;
			if (stateSum == 0) {
				newState = sumFixP;
			}
			else {
				//const float subSumF = toFloat(static_cast<FixedP2>(subSum));
				//const float stateSumF = toFloat(static_cast<FixedP>(stateSum));

				const float subSumF = static_cast<float>(subSum) / (1 << (N_BITS_DENOMINATOR * 2));
				const float stateSumF = static_cast<float>(stateSum) / (1 << N_BITS_DENOMINATOR);

				const float stimulusAdditionF = subSumF / stateSumF;

				newState = add_saturate(sumFixP, toFixedP(stimulusAdditionF));
			}
			write_2D_fixp(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newState);
		}

		template <int RADIUS>
		static void spStimulus_v0(
			const Image3D &stimulus,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const int numSamples,
			const bool ignoreMiddle)
		{
			throw 1;
			// last checked: 23-nov
			for (int x = 0; x < hiddenSummationTempBack._size.x; ++x) {
				for (int y = 0; y < hiddenSummationTempBack._size.y; ++y) {
					spStimulus_kernel<true, RADIUS>(x, y, stimulus, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, numSamples, ignoreMiddle);
				}
			}
		}

		template <int RADIUS>
		static void spStimulus_v1(
			const Image2D &visibleStates,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const bool ignoreMiddle)
		{
			std::tuple<int2, int2> ranges = cornerCaseRange(hiddenSummationTempBack._size, visibleStates._size, RADIUS, hiddenToVisible);
			const int x0 = 0;
			const int x1 = std::get<0>(ranges).x;
			const int x2 = std::get<0>(ranges).y;
			const int x3 = hiddenSummationTempBack._size.x;
			const int y0 = 0;
			const int y1 = std::get<1>(ranges).x;
			const int y2 = std::get<1>(ranges).y;
			const int y3 = hiddenSummationTempBack._size.y;

			for (int hiddenPosition_x = x0; hiddenPosition_x < x1; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					spStimulus_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					spStimulus_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, visibleStates, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, ignoreMiddle);
				}
			}
		}

		static void sfcStimulus(
			const Image3D &samples,
			const Image2D &hiddenSummationTempBack,
			Image2D &hiddenSummationTempFront, // write only
			const Image3D &weights,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const int radius,
			const int numSamples,
			const bool ignoreMiddle, 
			const int2 range)
		{
			// last checked: 25-nov 2016
			/*
			switch (radius) {
			case 6: spStimulus_v0<6>(stimulus, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, numSamples, ignoreMiddle); break;
			case 8: spStimulus_v0<8>(stimulus, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, numSamples, ignoreMiddle); break;
			case 20: spStimulus_v0<20>(stimulus, hiddenSummationTempBack, hiddenSummationTempFront, weights, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, numSamples, ignoreMiddle); break;
			default: printf("ERROR: SparseFeatures::spStimulus: provided radius %i is not implemented\n", radius); break;
			}
			*/

			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; ++hiddenPosition_y) {

					const int chunkPosition_x = hiddenPosition_x / chunkSize.x;
					const int chunkPosition_y = hiddenPosition_y / chunkSize.y;

					const int chunkCenter_x = getChunkCenter(chunkPosition_x, chunksToHidden.x);
					const int chunkCenter_y = getChunkCenter(chunkPosition_y, chunksToHidden.y);

					const int visiblePositionCenter_x = project(chunkCenter_x, hiddenToVisible.x);
					const int visiblePositionCenter_y = project(chunkCenter_y, hiddenToVisible.y);

					float subSum = 0.0f;
					int count = 0;

					const int fieldLowerBound_x = visiblePositionCenter_x - radius;
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					for (int s = 0; s < numSamples; ++s) {
						for (int dx = -radius; dx <= radius; dx++) {
							const int visiblePosition_x = visiblePositionCenter_x + dx;

							if (inBounds(visiblePosition_x, visibleSize.x)) {
								const int offset_x = visiblePosition_x - fieldLowerBound_x;

								for (int dy = -radius; dy <= radius; dy++) {
									const int visiblePosition_y = visiblePositionCenter_y + dy;

									if (inBounds(visiblePosition_y, visibleSize.y)) {
										const int offset_y = visiblePosition_y - fieldLowerBound_y;

										if (ignoreMiddle && (dx == 0) && (dy == 0)) {
											// do nothing
										} else {
											const int wi = s + numSamples * (offset_y + offset_x * (radius * 2 + 1));
											const float weight = read_3D(weights, hiddenPosition_x, hiddenPosition_y, wi);
											const float sample = read_3D(samples, visiblePosition_x, visiblePosition_y, s);
											const float delta = sample - weight;
											subSum += -delta * delta;
											count++;
										}
									}
								}
							}
						}
					}

					const float sum = read_2D(hiddenSummationTempBack, hiddenPosition_x, hiddenPosition_y);
					const float newValue = (count == 0) ? sum : sum + (subSum / count);
					//if (subSum > 0) std::cout << "INFO: SparseFeaturesChunk:sfcStimulus: subSum=" << subSum << "; count=" << count << "; newValue="<< newValue << std::endl;
					write_2D(hiddenSummationTempFront, hiddenPosition_x, hiddenPosition_y, newValue);
				}
			}
			//plots::plotImage(hiddenSummationTempFront, 4, "SparseFeaturesChunk:sfcStimulus");
		}

		static void sfcActivate(
			const Image2D &hiddenStimuli,
			const Array2D2f &hiddenStatesPrev,	// unused
			Array2D2f &hiddenActivationsFront,	// write only
			const int2 range)
		{
			// last checked: 28-nov 2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float hiddenStimulus = hiddenStimuli._data_float[i];
				const float newValue = exp(hiddenStimulus);
				hiddenActivationsFront._data_float[i] = { newValue, 0.0f };
			}
			//plots::plotImage(hiddenStimuli, 8, "SparseFeaturesChunk:sfcActivate:hiddenStimuli");
			//plots::plotImage(hiddenActivationsFront, 8, "SparseFeaturesChunk:sfcActivate;hiddenActivationsFront");
		}

		static void sfcInhibit(
			const Array2D2f &activations,
			Array2D2f &hiddenStatesFront,		// write only
			Array2Di2 &chunkWinners,		// write only
			const int2 hiddenSize,
			const int2 chunkSize,
			const int2 range)
		{
			//std::cout << "INFO: SparseFeaturesChunk:sfcInhibit: hiddenSize=" << hiddenSize.x << "," << hiddenSize.y << "; chunkSize=" << chunkSize.x << "," << chunkSize.y << "; range=" << range.x << "," << range.y << std::endl;

			// last checked: 28-nov 2016
			for (int chunkPosition_x = 0; chunkPosition_x < range.x; ++chunkPosition_x) {
				for (int chunkPosition_y = 0; chunkPosition_y < range.y; ++chunkPosition_y) {

					const int hiddenStartPosition_x = chunkPosition_x * chunkSize.x;
					const int hiddenStartPosition_y = chunkPosition_y * chunkSize.y;

					float maxValue = -99999.0f;
					int2 maxDelta = int2{ 0, 0 };

					for (int dx = 0; dx < chunkSize.x; ++dx) {
						const int hiddenPosition_x = hiddenStartPosition_x + dx;
						if (inBounds(hiddenPosition_x, hiddenSize.x)) {

							for (int dy = 0; dy < chunkSize.y; ++dy) {
								const int hiddenPosition_y = hiddenStartPosition_y + dy;
								if (inBounds(hiddenPosition_y, hiddenSize.y)) {

									const float activation = read_2D(activations, hiddenPosition_x, hiddenPosition_y).x;
									if (activation > maxValue) {
										maxValue = activation;
										maxDelta = int2{ dx, dy };
									}
								}
							}
						}
					}
					//if (maxValue == -99999.0f) std::cout << "WARNING:" << std::endl;
					write_2D(chunkWinners, chunkPosition_x, chunkPosition_y, maxDelta);

					for (int dx = 0; dx < chunkSize.x; ++dx) {
						for (int dy = 0; dy < chunkSize.y; ++dy) {
							const int hiddenPosition_x = hiddenStartPosition_x + dx;
							const int hiddenPosition_y = hiddenStartPosition_y + dy;

							if (inBounds(hiddenPosition_x, hiddenSize.x)) {
								if (inBounds(hiddenPosition_y, hiddenSize.y)) {

									//float tracePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).y;
									const float newValue = ((dx == maxDelta.x) && (dy == maxDelta.y)) ? 1.0f : 0.0f;
									write_2D(hiddenStatesFront, hiddenPosition_x, hiddenPosition_y, float2{ newValue, 0.0f });
								}
							}
						}
					}
				}
			}

			//plots::plotImage(activations, 8, "SparseFeaturesChunk:sfcInhibit:activations");
			//plots::plotImage(hiddenStatesFront, 8, "SparseFeaturesChunk:sfcInhibit:hiddenStatesFront");
		}

		static void sfcInhibitOther(
			const Array2D2f &activations,
			Array2D2f &hiddenStatesFront,		// write only
			const int2 hiddenSize,
			const int2 chunkSize,
			const int2 range)
		{
			// last checked: 25-nov 2016
			for (int chunkPosition_x = 0; chunkPosition_x < range.x; ++chunkPosition_x) {
				for (int chunkPosition_y = 0; chunkPosition_y < range.y; ++chunkPosition_y) {

					const int hiddenStartPosition_x = chunkPosition_x * chunkSize.x;
					const int hiddenStartPosition_y = chunkPosition_y * chunkSize.y;

					float maxValue = -99999.0f;
					int2 maxDelta = int2{ 0, 0 };

					for (int dx = 0; dx < chunkSize.x; ++dx) {
						const int hiddenPosition_x = hiddenStartPosition_x + dx;
						if (inBounds(hiddenPosition_x, hiddenSize.x)) {

							for (int dy = 0; dy < chunkSize.y; ++dy) {
								const int hiddenPosition_y = hiddenStartPosition_y + dy;
								if (inBounds(hiddenPosition_y, hiddenSize.y)) {

									const float activation = read_2D(activations, hiddenPosition_x, hiddenPosition_y).x;
									if (activation > maxValue) {
										maxValue = activation;
										maxDelta = int2{ dx, dy };
									}
								}
							}
						}
					}

					//if (maxValue == -99999.0f) std::cout << "WARNING: SparseFeaturesChunk: no activation is higher than " << maxValue << std::endl;

					for (int dx = 0; dx < chunkSize.x; ++dx) {
						const int hiddenPosition_x = hiddenStartPosition_x + dx;
						if (inBounds(hiddenPosition_x, hiddenSize.x)) {

							for (int dy = 0; dy < chunkSize.y; ++dy) {
								const int hiddenPosition_y = hiddenStartPosition_y + dy;
								if (inBounds(hiddenPosition_y, hiddenSize.y)) {

									//const float tracePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).y;
									const float newValue = ((dx == maxDelta.x) && (dy == maxDelta.y)) ? 1.0f : 0.0f;
									write_2D(hiddenStatesFront, hiddenPosition_x, hiddenPosition_y, { newValue, 0.0f });
								}
							}
						}
					}
				}
			}
			//plots::plotImage(activations, 8.0f, "SparseFeaturesChunk:sfcInhibitOther:activations");
			//plots::plotImage(hiddenStatesFront, 8.0f, "SparseFeaturesChunk:sfcInhibitOther:hiddenStatesFront");
		}

		template <bool CORNER, int RADIUS>
		static void sfcLearnWeights_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image2D &hiddenStates,
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const float weightAlpha,
			const int numSamples,
			const float gamma)
		{
			throw 1
			// last checked: 23-nov

#			ifdef USE_FIXED_POINT
			sfcLearnWeights_fixedp_kernel<CORNER, RADIUS>(
				hiddenPosition_x,
				hiddenPosition_y,
				hiddenStates,
				chunkWinners,
				samples,
				weightsBack,
				weightsFront,
				hiddenToVisible,
				hiddenSize,
				visibleSize,
				hiddenToVisible,
				chunkSize,
				chunksToHidden,
				weightAlpha,
				numSamples,
				gamma);
#			else
			sfcLearnWeights_floatp_kernel<CORNER, RADIUS>(
				hiddenPosition_x,
				hiddenPosition_y,
				hiddenStates,
				chunkWinners,
				samples,
				weightsBack,
				weightsFront,
				hiddenSize,
				visibleSize,
				hiddenToVisible,
				chunkSize,
				chunksToHidden,
				weightAlpha,
				numSamples,
				gamma);
#			endif
		}

		template <bool CORNER, int RADIUS>
		static void sfcLearnWeights_floatp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image2D &hiddenStates,
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const float weightAlpha,
			const int numSamples,
			const float gamma)
		{
			throw 1;
			// last checked: 23-nov

			const int chunkPosition_x = hiddenPosition_x / chunkSize.x;
			const int chunkPosition_y = hiddenPosition_y / chunkSize.y;

			const int chunkCenter_x = getChunkCenter(chunkPosition_x, chunksToHidden.x);
			const int chunkCenter_y = getChunkCenter(chunkPosition_y, chunksToHidden.y);

			const int visiblePositionCenter_x = project(chunkCenter_x, hiddenToVisible.x);
			const int visiblePositionCenter_y = project(chunkCenter_y, hiddenToVisible.y);

			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;

			const int2 chunkWinner = read_2D(chunkWinners, chunkPosition_x, chunkPosition_y);

			const int hiddenStartPosition_x = chunkPosition_x * chunkSize.x;
			const int hiddenStartPosition_y = chunkPosition_y * chunkSize.y;

			const int delta_x = (hiddenStartPosition_x + chunkWinner.x) - hiddenPosition_x;
			const int delta_y = (hiddenStartPosition_y + chunkWinner.y) - hiddenPosition_y;

			const float strength = exp(-(delta_x * delta_x + delta_y * delta_y) * gamma);

			//float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

			for (int s = 0; s < numSamples; ++s) {

				for (int dx = -RADIUS; dx <= RADIUS; dx++) {
					const int visiblePosition_x = visiblePositionCenter_x + dx;
					if (inBounds(visiblePosition_x, visibleSize.x)) {

						for (int dy = -RADIUS; dy <= RADIUS; dy++) {
							const int visiblePosition_y = visiblePositionCenter_y + dy;
							if (inBounds(visiblePosition_y, visibleSize.y)) {

								const int offset_x = visiblePosition_x - fieldLowerBound_x;
								const int offset_y = visiblePosition_y - fieldLowerBound_y;
								const int wi = s + numSamples * (offset_y + offset_x * (RADIUS * 2 + 1));

								const float weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
								const float sample = read_3D(samples, visiblePosition_x, visiblePosition_y, s);
								//float recon = read_imagef(recons, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;
								const float sLearn = strength * (sample - weightPrev);

								const float newValue = weightPrev + weightAlpha * sLearn;
								write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, newValue);
							}
						}
					}
				}
			}
		}

		template <bool CORNER, int RADIUS>
		static void sfcLearnWeights_fixedp_kernel(
			const int hiddenPosition_x,
			const int hiddenPosition_y,
			const Image2D &hiddenStates,
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const float weightAlpha,
			const int numSamples,
			const float gamma,
			const int2 range)
		{
			throw 1;
			//TODO
			/*
			const int visiblePositionCenter_x = project(hiddenPosition_x, hiddenToVisible.x);
			const int fieldLowerBound_x = visiblePositionCenter_x - RADIUS;
			const int fieldUpperBound_x = visiblePositionCenter_x + RADIUS;
			const int visiblePosStart_x = (CORNER) ? std::max(0, fieldLowerBound_x) : fieldLowerBound_x;
			const int visiblePosEnd_x = (CORNER) ? std::min(visibleSize.x, fieldUpperBound_x + 1) : fieldUpperBound_x + 1;

			const int visiblePositionCenter_y = project(hiddenPosition_y, hiddenToVisible.y);
			const int fieldLowerBound_y = visiblePositionCenter_y - RADIUS;
			const int fieldUpperBound_y = visiblePositionCenter_y + RADIUS;
			const int visiblePosStart_y = (CORNER) ? std::max(0, fieldLowerBound_y) : fieldLowerBound_y;
			const int visiblePosEnd_y = (CORNER) ? std::min(visibleSize.y, fieldUpperBound_y + 1) : fieldUpperBound_y + 1;

			const FixedP weightAlphaFixP = toFixedP(weightAlpha);
			if (weightAlphaFixP == 0) std::printf("WARNING: spLearnWeights_fixedp_kernel: weightAlphaFixP is zero\n");

			const FixedP hiddenState = read_2D_fixp(hiddenStates, hiddenPosition_x, hiddenPosition_y);
			const FixedP hiddenStateWeightAlpha = multiply_saturate(hiddenState, weightAlphaFixP);

			std::printf("INFO: spLearnWeights_fixedp_kernel: hiddenStateWeightAlpha=%i\n", static_cast<int>(hiddenStateWeightAlpha));


#			pragma ivdep
			for (int visiblePosition_x = visiblePosStart_x; visiblePosition_x < visiblePosEnd_x; ++visiblePosition_x) {
				const int offset_x = visiblePosition_x - fieldLowerBound_x;

#				pragma ivdep
				for (int visiblePosition_y = visiblePosStart_y; visiblePosition_y < visiblePosEnd_y; ++visiblePosition_y) {
					const int offset_y = visiblePosition_y - fieldLowerBound_y;

					const int wi = offset_y + (offset_x * ((RADIUS * 2) + 1));

					const FixedP weightPrev = read_3D_fixp(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
					const FixedP visibleState = read_2D_fixp(visibleStates, visiblePosition_x, visiblePosition_y);
					FixedP weight;
					if (visibleState > weightPrev)
					{
						const FixedP weightDelta = multiply_saturate(hiddenStateWeightAlpha, (visibleState - weightPrev));
						weight = add_saturate(weightPrev, weightDelta);
						//std::cout << "INFO: spLearnWeights_fixedp_kernel: A: weight_fixP=" << toFloat(weight) << "; weight_old=" << weight_old << std::endl;
					}
					else
					{
						const FixedP weightDelta = multiply_saturate(hiddenStateWeightAlpha, (weightPrev - visibleState));
						weight = substract_saturate(weightPrev, weightDelta);
						//std::cout << "INFO: spLearnWeights_fixedp_kernel: B: weight_fixP=" << toFloat(weight) << "; weight_old="<< weight_old << std::endl;
					}
					write_3D_fixp(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, weight);
				}
			}
			*/
		}

		template <int RADIUS>
		static void sfcLearnWeights_v1(
			const Image2D &hiddenStates,
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const float weightAlpha,
			const int numSamples,
			const float gamma,
			const int2 range)
		{
			throw 1;
			std::tuple<int2, int2> ranges = cornerCaseRange(hiddenSize, visibleSize, RADIUS, hiddenToVisible);
			const int x0 = 0;
			const int x1 = std::get<0>(ranges).x;
			const int x2 = std::get<0>(ranges).y;
			const int x3 = hiddenStates._size.x;
			const int y0 = 0;
			const int y1 = std::get<1>(ranges).x;
			const int y2 = std::get<1>(ranges).y;
			const int y3 = hiddenStates._size.y;

			for (int hiddenPosition_x = x0; hiddenPosition_x < x1; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
			}
			for (int hiddenPosition_x = x1; hiddenPosition_x < x2; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y1; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
				for (int hiddenPosition_y = y1; hiddenPosition_y < y2; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<false, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
				for (int hiddenPosition_y = y2; hiddenPosition_y < y3; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
			}
			for (int hiddenPosition_x = x2; hiddenPosition_x < x3; ++hiddenPosition_x) {
				for (int hiddenPosition_y = y0; hiddenPosition_y < y3; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<true, RADIUS>(hiddenPosition_x, hiddenPosition_y, hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
			}
		}

		template <int RADIUS>
		static void sfcLearnWeights_v0(
			const Image2D &hiddenStates,
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const float weightAlpha,
			const int numSamples,
			const float gamma,
			const int2 range) 
		{
			throw 1;
			for (int hiddenPosition_x = 0; hiddenPosition_x < hiddenStates._size.x; ++hiddenPosition_x) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < hiddenStates._size.y; ++hiddenPosition_y) {
					sfcLearnWeights_kernel<true, RADIUS>(hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma);
				}
			}
		}

		static void sfcLearnWeights(
			const Array2D2f &hiddenStates, // unused?
			const Array2Di2 &chunkWinners,
			const Image3D &samples,
			const Image3D &weightsBack,
			Image3D &weightsFront, // write only
			const int2 hiddenSize,
			const int2 visibleSize,
			const float2 hiddenToVisible,
			const int2 chunkSize,
			const float2 chunksToHidden,
			const int radius,
			const float weightAlpha,
			const int numSamples, 
			const float gamma,
			const int2 range)
		{
			/*
			switch (radius) {
			case 6: sfcLearnWeights_v1<6>(hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma, range); break;
			case 8: sfcLearnWeights_v1<8>(hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma, range); break;
			case 20: sfcLearnWeights_v1<20>(hiddenStates, chunkWinners, samples, weightsBack, weightsFront, hiddenSize, visibleSize, hiddenToVisible, chunkSize, chunksToHidden, weightAlpha, numSamples, gamma, range); break;
			default: printf("ERROR: SparseFeatures::sfcLearnWeights: provided radius %i is not implemented\n", radius); break;
			}
			*/
			// last checked: 28-nov 2016
			for (int hiddenPosition_x = 0; hiddenPosition_x < range.x; hiddenPosition_x++) {
				for (int hiddenPosition_y = 0; hiddenPosition_y < range.y; hiddenPosition_y++) {

					const int chunkPosition_x = hiddenPosition_x / chunkSize.x;
					const int chunkPosition_y = hiddenPosition_y / chunkSize.y;

					const int chunkCenter_x = getChunkCenter(chunkPosition_x, chunksToHidden.x);
					const int chunkCenter_y = getChunkCenter(chunkPosition_y, chunksToHidden.y);

					const int visiblePositionCenter_x = project(chunkCenter_x, hiddenToVisible.x);
					const int visiblePositionCenter_y = project(chunkCenter_y, hiddenToVisible.y);

					const int fieldLowerBound_x = visiblePositionCenter_x - radius;
					const int fieldLowerBound_y = visiblePositionCenter_y - radius;

					const int2 chunkWinner = read_2D(chunkWinners, chunkPosition_x, chunkPosition_y);

					const int hiddenStartPosition_x = chunkPosition_x * chunkSize.x;
					const int hiddenStartPosition_y = chunkPosition_y * chunkSize.y;

					const int delta_x = (hiddenStartPosition_x + chunkWinner.x) - hiddenPosition_x;
					const int delta_y = (hiddenStartPosition_y + chunkWinner.y) - hiddenPosition_y;

					const float strength = exp(-(delta_x * delta_x + delta_y * delta_y) * gamma);

					//float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

					for (int s = 0; s < numSamples; ++s) {

						for (int dx = -radius; dx <= radius; ++dx) {
							const int visiblePosition_x = visiblePositionCenter_x + dx;
							if (inBounds(visiblePosition_x, visibleSize.x)) {

								for (int dy = -radius; dy <= radius; ++dy) {
									const int visiblePosition_y = visiblePositionCenter_y + dy;
									if (inBounds(visiblePosition_y, visibleSize.y)) {

										const int offset_x = visiblePosition_x - fieldLowerBound_x;
										const int offset_y = visiblePosition_y - fieldLowerBound_y;
										const int wi = s + numSamples * (offset_y + offset_x * (radius * 2 + 1));

										const float weightPrev = read_3D(weightsBack, hiddenPosition_x, hiddenPosition_y, wi);
										const float sample = read_3D(samples, visiblePosition_x, visiblePosition_y, s);
										//float recon = read_imagef(recons, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;
										const float sLearn = strength * (sample - weightPrev);

										const float newValue = weightPrev + weightAlpha * sLearn;
										write_3D(weightsFront, hiddenPosition_x, hiddenPosition_y, wi, newValue);
									}
								}
							}
						}
					}
				}
			}
		}

		static void sfcLearnBiases(
			const Image2D &hiddenStimuli,
			const Image2D &hiddenStates,
			const Image2D &biasesBack,
			Image2D &biasesFront, //write only
			const float activeRatio, // unused
			const float biasAlpha,
			const int2 range)
		{
			// last checked: 25-nov 2016
			const int nElements = range.x * range.y;
#			pragma ivdep
			for (int i = 0; i < nElements; ++i) {
				const float hiddenState = hiddenStates._data_float[i];
				const float hiddenStimulus = hiddenStimuli._data_float[i];
				const float biasPrev = biasesBack._data_float[i];
				const float newValue = biasPrev + biasAlpha * (activeRatio - hiddenState);
				biasesFront._data_float[i] = newValue;
			}
		}

		static void sfcDeriveInputs(
			const Array2D2f &inputs,
			const Array2D2f &outputsBack,
			Array2D2f &outputsFront, // write only
			const float lambda,
			const int2 range)
		{
			// last checked: 28-nov 2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float input = inputs._data_float[i].x;
				const float tracePrev = outputsBack._data_float[i].y;

				const float newValueA = input;
				const float newValueB = lambda * tracePrev + (1.0f - lambda) * input;
				outputsFront._data_float[i] = float2{ newValueA, newValueB };
			}
		}
	};
}