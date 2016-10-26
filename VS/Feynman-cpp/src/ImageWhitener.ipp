// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.ipp"

namespace feynman {
	
	//Image whitener. Applies local whitening transformation to input.
	class ImageWhitener {
	private:

		//Resulting whitened image
		Image2D _result;

		//Size of the whitened image
		int2 _imageSize;

	public:

		/*!
		\brief Create the image whitener.
		Requires the image size and format, and ComputeProgram loaded with the extra kernel code (see ComputeProgram::loadExtraKernel).
		\param imageSize size of the source image (2D).
		\param imageFormat format of the image, e.g. CL_R, CL_RG, CL_RGB, CL_RGBA.
		\param imageType type of the image, e.g. CL_FLOAT, CL_UNORM_INT8.
		*/
		void create(
			int2 imageSize,
			int imageFormat,
			int imageType) 
		{
			_imageSize = imageSize;
			_result = Image2D(imageSize);
		}

		/*!
		\brief Filter (whiten) an image with a kernel radius
		\param cs is the ComputeSystem.
		\param input is the OpenCL 2D image to be whitened.
		\param kernelRadius local radius of examined pixels.
		\param intensity the strength of the whitening.
		*/
		void filter(
			Image2D &input,
			int kernelRadius,
			float intensity = 1024.0f) 
		{
			whiten(
				input,
				_result,
				_imageSize,
				kernelRadius,
				intensity,
				_imageSize);
		}

		//Return filtered image result
		const Image2D &getResult() const {
			return _result;
		}


	private:

		static void whiten(
			const Image2D &input, 
			Image2D &result,
			const int2 imageSize, 
			const int kernelRadius, 
			const float intensity,
			const int2 range) 
		{
			int2 position;
			for (int x = 0; x < range.x; ++x) {
				position.x = x;
				for (int y = 0; y < range.y; ++y) {
					position.y = y;

					const float currentColor = read_2D(input, position);
					float center = currentColor;
					int count = 0;

					for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
						for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
							if (dx == 0 && dy == 0)
								continue;

							const int2 otherPosition = position + int2{ dx, dy };

							if (inBounds(otherPosition, imageSize)) {
								float otherColor = read_2D(input, otherPosition);
								center = center + otherColor;
								count++;
							}
						}

					center = center / (count + 1);

					float4 centeredCurrentColor = currentColor - center;
					float4 covariances = (float4)(0.0f);

					for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
						for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
							if (dx == 0 && dy == 0)
								continue;

							int2 otherPosition = position + int2{ dx, dy };

							if (inBounds(otherPosition, imageSize)) {
								float4 otherColor = read_2D(input, otherPosition);
								float4 centeredOtherColor = otherColor - center;
								covariances = covariances + (centeredOtherColor * centeredCurrentColor);
							}
						}

					covariances = covariances / std::max(1, count);

					float centeredCurrentColorSigns = (centeredCurrentColor.x > 0.0f) ? 1.0f : -1.0f;

					// Modify color
					float whitenedColor = ((centeredCurrentColor > 0.0f) ? 1.0f : (float4)(-1.0f)) * (1.0f - exp(-fabs(intensity * covariances)));
					whitenedColor = std::min(1.0f, std::max(-1.0f, whitenedColor));

					write_2D(result, position, whitenedColor);

					float4 currentColor = read_2D(input, position);
					float4 center = currentColor;
					float count = 0.0f;

					for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
						for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
							if (dx == 0 && dy == 0)
								continue;

							int2 otherPosition = position + int2{ dx, dy };

							if (inBounds(otherPosition, imageSize)) {
								float4 otherColor = read_2D(input, otherPosition);
								center = center + otherColor;
								count++;
							}
						}

					center = center / (count + 1.0f);

					float4 centeredCurrentColor = currentColor - center;

					float4 covariances = (float4)(0.0f);

					for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
						for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
							if (dx == 0 && dy == 0)
								continue;

							int2 otherPosition = position + (int2)(dx, dy);

							if (inBounds(otherPosition, imageSize)) {
								float4 otherColor = read_2D(input, otherPosition);
								float4 centeredOtherColor = otherColor - center;
								covariances = covariances + centeredOtherColor * centeredCurrentColor;
							}
						}

					covariances = covariances / fmax(1.0f, count);

					float4 centeredCurrentColorSigns = (float4)(centeredCurrentColor.x > 0.0f ? 1.0f : -1.0f,
						centeredCurrentColor.y > 0.0f ? 1.0f : -1.0f,
						centeredCurrentColor.z > 0.0f ? 1.0f : -1.0f,
						centeredCurrentColor.w > 0.0f ? 1.0f : -1.0f);

					// Modify color
					float4 whitenedColor = fmin(1.0f, fmax(-1.0f, (centeredCurrentColor > 0.0f ? (float4)(1.0f) : (float4)(-1.0f)) * (1.0f - exp(-fabs(intensity * covariances)))));

					write_2D(result, position, whitenedColor);
				}
			}
		}
	};
}