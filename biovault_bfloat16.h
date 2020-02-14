/*******************************************************************************
* Copyright 2020 LKEB, Leiden University Medical Center
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Adapted from the original dnnl::impl::bfloat16_t implementation by Intel Corporation,
// which is licensed under the Apache License, Version 2.0:
// https://github.com/intel/mkl-dnn/blob/v1.2/LICENSE

#include <cinttypes>
#include <cmath>
#include <cfloat>
#include <cstring>


namespace biovault {

	class bfloat16_t {

	private:
		uint16_t raw_bits_;

	public:
		bfloat16_t() = default;
		constexpr bfloat16_t(uint16_t r, bool) : raw_bits_(r) {}
		bfloat16_t(float f) { (*this) = f; }

		bfloat16_t &operator=(float f);

		operator float() const;

		bfloat16_t &operator+=(bfloat16_t a) {
			(*this) = (float)(*this) + (float)a;
			return *this;
		}
	};

	static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

	inline bfloat16_t &bfloat16_t::operator=(const float f) {
			uint16_t iraw[2];
			std::memcpy(iraw, &f, sizeof(float));

			switch (std::fpclassify(f)) {
			case FP_SUBNORMAL:
			case FP_ZERO:
				// sign preserving zero (denormal go to zero)
				raw_bits_ = iraw[1];
				raw_bits_ &= 0x8000;
				break;
			case FP_INFINITE: raw_bits_ = iraw[1]; break;
			case FP_NAN:
				// truncate and set MSB of the mantissa force QNAN
				raw_bits_ = iraw[1];
				raw_bits_ |= 1 << 6;
				break;
			case FP_NORMAL:
				// round to nearest even and truncate
				unsigned int rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
				uint32_t int_raw;
				std::memcpy(&int_raw, &f, sizeof(float));
				int_raw += rounding_bias;
				std::memcpy(iraw, &int_raw, sizeof(float));
				raw_bits_ = iraw[1];
				break;
		}
		return *this;
	}

	inline bfloat16_t::operator float() const {
		const uint16_t iraw[2] = { 0, raw_bits_ };
		float f;
		std::memcpy(&f, iraw, sizeof(float));
		return f;
	}

}
