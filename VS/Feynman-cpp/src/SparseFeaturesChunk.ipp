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
			DoubleBuffer2D<float> _derivedInput;

			// Samples (time sliced derived inputs)
			DoubleBuffer3D<float> _samples;

			//Reconstruction errors
			Array3D<float> _recons;

			//Weights
			DoubleBuffer3D<float> _weights; // Encoding weights (creates spatio-temporal sparse code)

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

			std::string info() const override {
				std::string result = "\n";
				for (size_t i = 0; i < _visibleLayerDescs.size(); ++i) {
					result += "visibleLayer[" + std::to_string(i) + "]: size=(" + std::to_string(_visibleLayerDescs[i]._size.x) + "," + std::to_string(_visibleLayerDescs[i]._size.y) + ")\n";
					result += "visibleLayer[" + std::to_string(i) + "]: radius=" + std::to_string(_visibleLayerDescs[i]._radius) + "\n";
					result += "visibleLayer[" + std::to_string(i) + "]: weightAlpha=" + std::to_string(_visibleLayerDescs[i]._weightAlpha) + "\n";
					result += "visibleLayer[" + std::to_string(i) + "]: ignoreMiddle=" + std::to_string(_visibleLayerDescs[i]._ignoreMiddle) + "\n";
				}
				result += "TODO\n";
/*				result += "hiddenSize=(" + std::to_string(_hiddenSize.x) + "," + std::to_string(_hiddenSize.y) + ")\n";;
				result += "inhibitionRadius=" + std::to_string(_inhibitionRadius) + "\n";
				result += "activeRatio=" + std::to_string(_activeRatio) + "\n";
				result += "biasAlpha=" + std::to_string(_biasAlpha) + "\n";
				result += "initWeightRange=(" + std::to_string(_initWeightRange.x) + "," + std::to_string(_initWeightRange.y) + ")\n";
				result += "initBiasRange=(" + std::to_string(_initBiasRange.x) + "," + std::to_string(_initBiasRange.y) + ")\n";
*/				return result;
			}

			//Factory
			std::shared_ptr<SparseFeatures> sparseFeaturesFactory() override {
				if (true) std::cout << "INFO: SFChunk:sparseFeaturesFactory:" << info() << std::endl;
				return std::make_shared<SparseFeaturesChunk>(_visibleLayerDescs, _hiddenSize, _chunkSize, _numSamples, _biasAlpha, _gamma, _initWeightRange, _rng);
			}
		};

	private:

		//Activations, states, biases
		DoubleBuffer2D<float> _hiddenStates;
		DoubleBuffer2D<float> _hiddenActivations;
		DoubleBuffer2D<float> _hiddenBiases;
		Array2D<int2> _chunkWinners;

		int2 _hiddenSize;

		//Ratio between number of hidden states and number of chunks
		float2 _chunkToHidden;

		//Size of chunks
		int2 _chunkSize;

		//Hidden summation temporary buffer
		DoubleBuffer2D<float> _hiddenSummationTemp;

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
					vl._weights = createDoubleBuffer3D<float>(weightsSize);

					//std::cout << "INFO: SparseFeaturesChunk:constructor: initWeightRange=" << initWeightRange.x << "," << initWeightRange.y << std::endl;
					randomUniform3D(vl._weights[_back], initWeightRange, rng);
					//plots::plotImage(vl._weights[_back], 6, "SFChunk:constructor:weights" + std::to_string(vli));
				}
				vl._derivedInput = createDoubleBuffer2D<float>(vld._size);
				clear(vl._derivedInput[_back]);

				vl._samples = createDoubleBuffer3D<float>({ vld._size.x, vld._size.y, numSamples });
				clear(vl._samples[_back]);

				//vl._recons = Array3D({vld._size.x, vld._size.y, numSamples});
				//clear(vl._recons);
			}

			// Hidden state data
			_hiddenStates = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenActivations = createDoubleBuffer2D<float>(_hiddenSize);
			_hiddenBiases = createDoubleBuffer2D<float>(_hiddenSize);
			_chunkWinners = Array2D<int2>(int2{ chunksInX, chunksInY });
			_hiddenSummationTemp = createDoubleBuffer2D<float>(_hiddenSize);

			clear(_hiddenStates[_back]);
			clear(_hiddenActivations[_back]);

			if (true) {
				randomUniform2D(_hiddenBiases[_back], initWeightRange, rng);
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
			const std::vector<Array2D<float>> &visibleStates,
			const Array2D<float> &predictionsPrev,
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

				// Derive inputs
				if (EXPLAIN) std::cout << "EXPLAIN: SFChunk:activate: visible layer " << vli << "/" << _visibleLayers.size() << ": deriving inputs." << std::endl;
				sfcDeriveInputs(
					visibleStates[vli],			// in
					vl._derivedInput[_back],	// unused
					vl._derivedInput[_front],	// out
					vld._lambda,
					vld._size
				);

				//plots::plotImage(vl._derivedInput[_front], 6, "SFChunk:activate:derivedInput" + std::to_string(vli));

				// Add sample
				if (EXPLAIN) std::cout << "EXPLAIN: SFChunk:activate: visible layer " << vli << "/" << _visibleLayers.size() << ": sampling inputs (" << _numSamples << ")." << std::endl;
				sfcAddSample(
					vl._derivedInput[_front],		// in
					vl._samples[_back],				// in
					vl._samples[_front],			// out: 
					_numSamples,
					vld._size
				);

				//plots::plotImage(vl._samples[_front], 6, "SFChunk:activate:samples" + std::to_string(vli));
				//plots::plotImage(vl._weights[_back], 6, "SFChunk:activate:weights" + std::to_string(vli));

				if (EXPLAIN) std::cout << "EXPLAIN: SFChunk:activate: visible layer " << vli << "/" << _visibleLayers.size() << ": adding inputs to stimuls influx." << std::endl;
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
			if (EXPLAIN) std::cout << "EXPLAIN: SFChunk:activate: calculating activation based on stimulus influx." << std::endl;
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

			if (EXPLAIN) std::cout << "EXPLAIN: SFChunk:activate: calculating hidden state SDR based on activations." << std::endl;
			sfcInhibit(
				_hiddenActivations[_front],		// in
				_hiddenStates[_front],			// out: note: _hiddenStates are used in learn
				_chunkWinners,					// out: note: _chunkWinners are used in learn
				_hiddenSize,
				_chunkSize,
				int2{ chunksInX , chunksInY }
			);
			plots::plotImage(_hiddenActivations[_front], DEBUG_IMAGE_WIDTH, "SFChunk:activate:hiddenActivations");
			plots::plotImage(_hiddenStates[_front], DEBUG_IMAGE_WIDTH, "SFChunk:activate:hiddenStates");
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

				//std::cout << "INFO: SFChunk:learn: weightAlpha=" << vld._weightAlpha << "; gamma="<< _gamma <<std::endl;

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

			// Learn biases
			/*{
			float activeRatio = 1.0f / (_chunkSize.x * _chunkSize.y);

			int argIndex = 0;

			_learnBiasesKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_learnBiasesKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
			_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
			_learnBiasesKernel.setArg(argIndex++, activeRatio);
			_learnBiasesKernel.setArg(argIndex++, _biasAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
			}*/
		}

		/*!
		\brief Inhibit
		Inhibits given activations using this encoder's inhibitory pattern
		*/
		void inhibit(
			const Array2D<float> &activations,
			Array2D<float> &states,
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
		const DoubleBuffer2D<float> &getHiddenStates() const override {
			// last checked: 25-nov
			return _hiddenStates;
		}

		/*!
		\brief Get context
		*/
		const Array2D<float> &getHiddenContext() const override {
			// note: returning activations, not an sdr
			return _hiddenActivations[_back];
			//return _hiddenStates[_back]; // return an sdr
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

	private:

		static void sfcAddSample(
			const Array2D<float> &visibleStates,
			const Array3D<float> &samplesBack,
			Array3D<float> &samplesFront,
			const int numSamples,
			const int2 range)
		{
			// last checked: 25-nov 2016
			for (int position_x = 0; position_x < range.x; ++position_x) {
				for (int position_y = 0; position_y < range.y; ++position_y) {
					const float visibleState = read_2D(visibleStates, position_x, position_y);
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
			const Array3D<float> &stimulus,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
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

		static void sfcStimulus(
			const Array3D<float> &samples,
			const Array2D<float> &hiddenSummationTempBack,
			Array2D<float> &hiddenSummationTempFront, // write only
			const Array3D<float> &weights,
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
			const Array2D<float> &hiddenStimuli,	// in
			const Array2D<float> &hiddenStatesPrev,	// unused
			Array2D<float> &hiddenActivationsFront,	// write only
			const int2 range)
		{
			// last checked: 28-nov 2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float hiddenStimulus = hiddenStimuli._data_float[i];
				const float newValue = exp(hiddenStimulus);
				hiddenActivationsFront._data_float[i] = newValue;
			}
			//plots::plotImage(hiddenStimuli, 8, "SparseFeaturesChunk:sfcActivate:hiddenStimuli");
			//plots::plotImage(hiddenActivationsFront, 8, "SparseFeaturesChunk:sfcActivate;hiddenActivationsFront");
		}

		static void sfcInhibit(
			const Array2D<float> &activations,	// in
			Array2D<float> &hiddenStatesFront,	// write only
			Array2D<int2> &chunkWinners,		// write only
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

									const float activation = read_2D(activations, hiddenPosition_x, hiddenPosition_y);
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
									write_2D(hiddenStatesFront, hiddenPosition_x, hiddenPosition_y, newValue);
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
			const Array2D<float> &activations,
			Array2D<float> &hiddenStatesFront,		// write only
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

									const float activation = read_2D(activations, hiddenPosition_x, hiddenPosition_y);
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
									write_2D(hiddenStatesFront, hiddenPosition_x, hiddenPosition_y, newValue);
								}
							}
						}
					}
				}
			}
			//plots::plotImage(activations, 8.0f, "SparseFeaturesChunk:sfcInhibitOther:activations");
			//plots::plotImage(hiddenStatesFront, 8.0f, "SparseFeaturesChunk:sfcInhibitOther:hiddenStatesFront");
		}

		static void sfcLearnWeights(
			const Array2D<float> &hiddenStates, // unused?
			const Array2D<int2> &chunkWinners,
			const Array3D<float> &samples,
			const Array3D<float> &weightsBack,
			Array3D<float> &weightsFront, // write only
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
			//std::cout << "INFO: SFChunk:sfcLearnWeights: weightAlpha=" << weightAlpha << "; gamma="<< gamma << std::endl;

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
			const Array2D<float> &hiddenStimuli,
			const Array2D<float> &hiddenStates,
			const Array2D<float> &biasesBack,
			Array2D<float> &biasesFront, //write only
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
			const Array2D<float> &inputs,
			const Array2D<float> &outputsBack,
			Array2D<float> &outputsFront, // write only
			const float lambda,
			const int2 range)
		{
			if (lambda != 0.0f) std::cout << "WARNING: SFChunk: sfcDeriveInputs: lambda=" << lambda << std::endl;

			// last checked: 28-nov 2016
			const int nElements = range.x * range.y;
			for (int i = 0; i < nElements; ++i) {
				const float input = inputs._data_float[i];
				//const float tracePrev = outputsBack._data_float[i].y;
				const float newValueA = input;
				//const float newValueB = lambda * tracePrev + (1.0f - lambda) * input;
				//outputsFront._data_float[i] = float2{ newValueA, newValueB };
				outputsFront._data_float[i] = newValueA;
			}
		}
	};
}