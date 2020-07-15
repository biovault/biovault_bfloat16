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

// Initial author: Niels Dekker (LKEB)

// The file to be tested. Included twice here, to check its include guards!
#include "biovault_bfloat16.h"
#include "biovault_bfloat16.h"

// GoogleTest header file:
#include <gtest/gtest.h>

// Standard library header files:
#include <array>
#include <cmath>    // For fpclassify.
#include <cstring>
#include <string>
#include <limits>

// References:
//
// Intel, "BFLOAT16 – Hardware Numerics Definition", White Paper, November 2018, Revision 1.0
// Document Number: 338302-001US 
// https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf
// https://software.intel.com/en-us/download/bfloat16-hardware-numerics-definition
//
// Wikipedia "bfloat16 floating-point format"
// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
//
// John D. Cook, 15 November 2018,
// "Comparing bfloat16 range and precision to other 16-bit numbers"
// https://www.johndcook.com/blog/2018/11/15/bfloat16


namespace
{
#ifndef BIOVAULT_BFLOAT16_EXHAUSTIVE_TEST
#  if defined(NDEBUG)
#  define BIOVAULT_BFLOAT16_EXHAUSTIVE_TEST 1
#  else
#  define BIOVAULT_BFLOAT16_EXHAUSTIVE_TEST 0
#  endif
#endif
	constexpr bool exhaustive{ BIOVAULT_BFLOAT16_EXHAUSTIVE_TEST };

	using biovault::bfloat16_t;

	using float_limits = std::numeric_limits<float>;
	using array_of_bytes = std::array<std::uint8_t, sizeof(float)>;

	constexpr auto uint16_max = std::numeric_limits<std::uint16_t>::max();


	// Return the four bytes that the specified float consists of.
	array_of_bytes float_to_array_of_bytes(const float arg)
	{
		array_of_bytes result;
		std::memcpy(result.data(), &arg, sizeof(float));
		return result;
	}


	// Note: Conversion from 32-bit float to bfloat16 may not be lossless!
	bfloat16_t float_to_bfloat16(const float arg)
	{
		return bfloat16_t(arg);
	}


	bfloat16_t raw_bits_to_bfloat16(const std::uint16_t arg)
	{
#ifdef BIOVAULT_BFLOAT16_TEST_NO_UINT16_AND_BOOL_CONSTRUCTOR
		bfloat16_t result;
		std::memcpy(&result, &arg, sizeof(arg));
		return result;
#else
		return bfloat16_t(arg, true);
#endif
	}


	// Note: Conversion from 32-bit float to uint16 may not be lossless!
	std::uint16_t float_to_raw_bits_of_bfloat16(const float arg)
	{
		return get_raw_bits(float_to_bfloat16(arg));
	}


	// Do float --> bfloat16 --> float
	float roundtrip_float(const float arg)
	{
		return float_to_bfloat16(arg);
	}


	// Assert that float --> bfloat16 --> float is lossless, for the specified argument.
	void assert_lossless_roundtrip(const float expected_float)
	{
		const float actual_float = roundtrip_float(expected_float);

		EXPECT_EQ(float_to_array_of_bytes(actual_float), float_to_array_of_bytes(expected_float));

		const auto expected_float_category = std::fpclassify(expected_float);

		ASSERT_EQ(std::fpclassify(actual_float), expected_float_category);

		if (expected_float_category != FP_NAN)
		{
			ASSERT_EQ(actual_float, expected_float);
		}
	}


	template <typename T>
	void assert_conversion_from_specified_value_equals_conversion_from_float(const T value)
	{
		ASSERT_EQ(
			get_raw_bits(bfloat16_t{ value }),
			get_raw_bits(bfloat16_t{ static_cast<float>(value) }));
	}

	template <typename T>
	void assert_conversion_from_integer_type_equals_conversion_from_float()
	{
		// Tests all values of the integer type, from its min to its max.
		constexpr auto max_value = std::numeric_limits<T>::max();

		for (auto i = std::numeric_limits<T>::min(); i < max_value; ++i)
		{
			assert_conversion_from_specified_value_equals_conversion_from_float(i);
		}
		assert_conversion_from_specified_value_equals_conversion_from_float(max_value);
	}

	template <typename T>
	void assert_conversion_from_min_and_max_equals_conversion_from_float()
	{
		assert_conversion_from_specified_value_equals_conversion_from_float(std::numeric_limits<T>::min());
		assert_conversion_from_specified_value_equals_conversion_from_float(std::numeric_limits<T>::max());
	}


	template <typename T>
	void assert_assignment_yields_same_raw_bits_as_construction_from_value(const T value)
	{
		bfloat16_t bf16;
		// Assignment to be tested:
		bf16 = value;
		ASSERT_EQ(get_raw_bits(bf16), get_raw_bits(bfloat16_t{ value }));
	}


	template <typename T>
	void assert_assignment_from_integer_type_yields_same_raw_bits_as_construction_from_value()
	{
		// Tests all values of the integer type, from its min to its max.
		constexpr auto max_value = std::numeric_limits<T>::max();

		for (auto i = std::numeric_limits<T>::min(); i < max_value; ++i)
		{
			assert_assignment_yields_same_raw_bits_as_construction_from_value(i);
		}
		assert_assignment_yields_same_raw_bits_as_construction_from_value(max_value);
	}

	template <typename T>
	void assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value()
	{
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::numeric_limits<T>::min());
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::numeric_limits<T>::max());
	}

}


GTEST_TEST(bfloat16, EightBitWholeNumberRoundTripIsLossless)
{
	// bfloat16_t has only 7 bits for its mantissa, but it implicitly has a 1 as the most significant bit.

	// For i = 256 down to zero.
	for (std::int16_t i{ std::uint8_t{ 1 } << std::uint8_t{8} }; i >= 0; --i)
	{
		assert_lossless_roundtrip(static_cast<float>(i));
		assert_lossless_roundtrip(-static_cast<float>(i));
	}
}


GTEST_TEST(bfloat16, PowerOfTwoRoundTripIsLossless)
{
	// For exponent = 128 down to one.
	for (std::uint8_t exponent{ std::uint8_t{1u} << std::uint8_t{7u} }; exponent > 0; --exponent)
	{
		SCOPED_TRACE(std::string("exponent") + std::to_string(exponent));
		assert_lossless_roundtrip(std::pow(2.0f, int{ exponent }));
	}

	// "The minimum positive normal value is 2 ^ −126..."
	// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding

	constexpr auto abs_exponent_of_minimum_positive = (std::uint8_t{ 1u } << std::uint8_t{ 7u }) - std::uint8_t{ 2 };

	ASSERT_EQ(std::pow(2.0f, -int{ abs_exponent_of_minimum_positive }), float_limits::min());

	// For exponent = -126 up to minus one.
	for (std::uint8_t abs_exponent{ abs_exponent_of_minimum_positive }; abs_exponent > 0; --abs_exponent)
	{
		SCOPED_TRACE(std::string("exponent -") + std::to_string(abs_exponent));
		assert_lossless_roundtrip(std::pow(2.0f, -int{ abs_exponent }));
	}
}


GTEST_TEST(bfloat16, MaxBFloat16RoundTripIsLossless)
{
	constexpr auto max_bfloat16 = 3.38953139e38f;
	constexpr auto max_float32 = 3.402823466e38f;

	ASSERT_EQ(max_float32, float_limits::max());
	ASSERT_LT(max_bfloat16, float_limits::max());

	// "The maximum positive finite value of a normal bfloat16 number is 3.38953139 x 10^38,
	// slightly below (2^24 − 1) x 2^−23 x 2^127 = 3.402823466  x 10^38, the max finite
	// positive value representable in single precision. 
	// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples

	assert_lossless_roundtrip(max_bfloat16);
}


GTEST_TEST(bfloat16, NanInfinityMinAndEpsilonRoundTripsAreLossless)
{
	assert_lossless_roundtrip(float_limits::quiet_NaN());
	assert_lossless_roundtrip(float_limits::infinity());
	assert_lossless_roundtrip(-float_limits::infinity());
	assert_lossless_roundtrip(float_limits::min());
	assert_lossless_roundtrip(-float_limits::min());
	assert_lossless_roundtrip(float_limits::epsilon());
	assert_lossless_roundtrip(-float_limits::epsilon());
}


GTEST_TEST(bfloat16, MaxAndLowestFloatsConvertToInfinity)
{
	ASSERT_EQ(roundtrip_float(float_limits::max()), float_limits::infinity());
	ASSERT_EQ(roundtrip_float(-float_limits::max()), -float_limits::infinity());
	ASSERT_EQ(roundtrip_float(float_limits::lowest()), -float_limits::infinity());
}


GTEST_TEST(bfloat16, DenormalFloatsConvertToZero)
{
	// As proposed by https://software.intel.com/en-us/download/bfloat16-hardware-numerics-definition
	// page 6, "1.2.1 FMA Unit":
	// * Treat denormal source as zero by default (only this mode is supported).
	// * Flush denormal results to zero by default (only this mode is supported).

	constexpr auto zeroArray = array_of_bytes{};
	const auto minusZeroArray = float_to_array_of_bytes(-0.0f);

	EXPECT_EQ(roundtrip_float(float_limits::min() / 2.0f), 0.0f);
	EXPECT_EQ(roundtrip_float(-float_limits::min() / 2.0f), 0.0f);

	EXPECT_EQ(roundtrip_float(float_limits::denorm_min()), 0.0f);
	EXPECT_EQ(roundtrip_float(-float_limits::denorm_min()), 0.0f);

	const auto denom_max = nextafterf(float_limits::min(), 0.0f);
	ASSERT_LT(denom_max, float_limits::min());
	ASSERT_GT(denom_max, 0.0f);

	EXPECT_EQ(roundtrip_float(denom_max), 0.0f);
	EXPECT_EQ(roundtrip_float(-denom_max), 0.0f);

	if (exhaustive)
	{
		// Might take a few seconds!
		for (auto denorm = denom_max; denorm > 0.0f; denorm = std::nextafterf(denorm, 0.0f))
		{
			EXPECT_EQ(float_to_array_of_bytes(roundtrip_float(denorm)), zeroArray);
			EXPECT_EQ(float_to_array_of_bytes(roundtrip_float(-denorm)), minusZeroArray);
		}
	}
}

GTEST_TEST(bfloat16, Epsilon)
{
	// According to John D. Cook,
	// Comparing bfloat16 range and precision to other 16-bit numbers:
	//
	// |--------+------------|
	// | Format |    Epsilon |
	// |--------+------------|
	// | FP32   | 0.00000012 |
	// | FP16   | 0.00097656 |
	// | BF16   | 0.00781250 |
	// |--------+------------|
	// https://www.johndcook.com/blog/2018/11/15/bfloat16/
	// (Web page checked on July 8, 2020.) 

	if (exhaustive)
	{
		const auto next_bfloat16 = []
		{
			const bfloat16_t bfloat16_one{ 1 };

			auto f = 0.0f;

			do
			{
				f = std::nextafterf(f, 1.0f);
			} while (bfloat16_t{ 1.0f + f } <= bfloat16_one);

			return bfloat16_t{ 1.0f + f };
		}();

		EXPECT_GT(float{ next_bfloat16 }, 1.0f);

		const bfloat16_t bfloat16_epsilon{ float{next_bfloat16} - 1.0f };

		EXPECT_GT(float{ bfloat16_epsilon }, float_limits::epsilon());
		EXPECT_LT(float{ bfloat16_epsilon }, 1.0f);
		EXPECT_EQ(float{ bfloat16_epsilon }, 0.00781250f);
	}
}

#ifndef BIOVAULT_BFLOAT16_TEST_NO_UINT16_AND_BOOL_CONSTRUCTOR
GTEST_TEST(bfloat16, AllowsConstexprConstructionFromRawBits)
{
	constexpr bfloat16_t bfloat16_from_raw_bits(std::uint16_t{}, bool{});
	const float f = bfloat16_from_raw_bits;
	EXPECT_EQ(f, 0.0f);
}
#endif


GTEST_TEST(bfloat16, RawBitsRoundTripIsLossless)
{
	ASSERT_EQ(get_raw_bits(raw_bits_to_bfloat16(std::uint16_t{})), std::uint16_t{});

	for (auto i = uint16_max; i > 0; --i)
	{
		ASSERT_EQ(get_raw_bits(raw_bits_to_bfloat16(i)), i);
	}
}


GTEST_TEST(bfloat16, RawBitsOfInfinityConvertToCorrespondingFloat)
{
	// From Wikipedia:
	//
	// Just as in IEEE 754, positive and negative infinity are represented with
	// their corresponding sign bits, all 8 exponent bits set (FF hex) and all
	// significand bits zero. Explicitly,
	//
	//   val    s_exponent_signcnd
	//	 +inf = 0_11111111_0000000
	//	 -inf = 1_11111111_0000000
	// 
	// Source: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Positive_and_negative_infinity 

	constexpr auto positive_infinity_bits = 0b0'11111111'0000000;
	constexpr auto negative_infinity_bits = 0b1'11111111'0000000;

	ASSERT_EQ(float{ raw_bits_to_bfloat16(positive_infinity_bits) }, float_limits::infinity());
	ASSERT_EQ(float{ raw_bits_to_bfloat16(negative_infinity_bits) }, -float_limits::infinity());
}


GTEST_TEST(bfloat16, RawBitsOfNanConvertToNanFloat)
{
	// From Wikipedia:
	//
	// Just as in IEEE 754, NaN values are represented with either sign bit, all
	// 8 exponent bits set (FF hex) and not all significand bits zero. Explicitly,
	//
	//   val    s_exponent_signcnd
	//   +NaN = 0_11111111_klmnopq
	//   -NaN = 1_11111111_klmonpq
	//
	// where at least one of k, l, m, n, o, p, or q is 1.
	//
	// Source: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Not_a_Number

	constexpr std::uint16_t exponent_bits{ 0b0'11111111'0000000 };
	constexpr std::uint16_t significand_bits{ 0b1111111 };

	for (auto bits = uint16_max; bits > 0; --bits)
	{
		const bool are_all_exponent_bits_set = (bits & exponent_bits) == exponent_bits;
		const bool are_all_significand_bits_zero = (bits & significand_bits) == 0;

		ASSERT_EQ(
			std::isnan(float{ raw_bits_to_bfloat16(bits)}),
			are_all_exponent_bits_set && !are_all_significand_bits_zero);
	}
	ASSERT_FALSE(std::isnan(float{ raw_bits_to_bfloat16(0) }));
}


GTEST_TEST(bfloat16, RawBitsOfNormalConvertToNormalFloat)
{
	// From Wikipedia:
	//
	// -----------------------------------------------------------
	// |    Exponent   | Significand zero | Significand non-zero
	// -----------------------------------------------------------
	// |      00H      |     zero, −0     |   subnormal numbers
	// | 01H, ..., FEH |          normalized value
	// |      FFH      |    +/-infinity   | NaN (quiet, signaling)
	// -----------------------------------------------------------
	//
	// Source: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding 

	for (auto bits = uint16_max; bits > 0; --bits)
	{
		const auto exponent = (bits >> 7U) & 0xFF;
		ASSERT_EQ(
			std::isnormal(float{ raw_bits_to_bfloat16(bits) }),
			(exponent > 0) && (exponent < 0xFF));
	}
	ASSERT_FALSE(std::isnormal(float{ raw_bits_to_bfloat16(0) }));
}


GTEST_TEST(bfloat16, NormalBFloat16ToFloatRoundTripIsLossLess)
{
	for (auto bits = uint16_max; bits > 0; --bits)
	{
		const float f{ raw_bits_to_bfloat16(bits) };

		if (std::isnormal(f))
		{
			ASSERT_EQ(get_raw_bits(bfloat16_t{ f }), bits);
		}
	}
}


GTEST_TEST(bfloat16, MostSignificantRawBitConvertsToSignBitOfFloat)
{
	for (auto bits = uint16_max; bits > 0; --bits)
	{
		ASSERT_EQ(
			std::signbit(float{ raw_bits_to_bfloat16(bits) }),
			(bits & (1U << 15U)) != 0);
	}
	ASSERT_FALSE(std::signbit(float{ raw_bits_to_bfloat16(0) }));
}


GTEST_TEST(bfloat16, RawRoundTrip)
{
	constexpr std::uint16_t _15{ std::numeric_limits<std::uint16_t>::digits - 1 };
	constexpr std::uint16_t _64{ 1u << 6u };
	constexpr std::uint16_t _128{ 1u << 7u };
	constexpr std::uint16_t _32768{ 1u << _15 };

	constexpr std::uint16_t _32641{ 0x7f81 };
	constexpr std::uint16_t _32703{ 0x7fbf };
	constexpr std::uint16_t _65409{ 0xff81 };
	constexpr std::uint16_t _65471{ 0xffbf };

	static_assert((_15 == 15) && (_64 == 64) && (_128 == 128) && (_32641 == 32641) && (_32703 == 32703) && (_32768 == 32768) && (_65409 == 65409) && (_65471 == 65471),
		"Magic number check");

	const float zero_float{ raw_bits_to_bfloat16(std::uint16_t{}) };
	EXPECT_EQ(zero_float, 0.0f);
	EXPECT_EQ(float_to_array_of_bytes(zero_float), array_of_bytes());
	assert_lossless_roundtrip(zero_float);

	const auto raw_bits_of_minus_zero_bfloat16 = float_to_raw_bits_of_bfloat16(-0.0f);

	for (auto i = uint16_max; i > 0; --i)
	{
		SCOPED_TRACE(std::to_string(i));
		const auto initial_bfloat16 = raw_bits_to_bfloat16(i);
		const float f{ initial_bfloat16 };
		const auto float_category = std::fpclassify(f);

		if ((i & ~_32768) < _128)
		{
			if (i == _32768)
			{
				// i has the raw bits of minus zero.
				ASSERT_EQ(float_category, FP_ZERO);
				ASSERT_TRUE(std::signbit(f));
				assert_lossless_roundtrip(f);
			}
			else
			{
				// i has the raw bits of a denormal.
				ASSERT_EQ(float_category, FP_SUBNORMAL);

				// In this case, i -> bfloat16 -> float -> bfloat16 yields either +0.0f or -0.0f.
				EXPECT_EQ(float_to_raw_bits_of_bfloat16(f), (i < _128) ? std::uint16_t{} : raw_bits_of_minus_zero_bfloat16);
			}
		}
		else
		{
			const auto roundtripped_bfloat = float_to_bfloat16(f);
			const float round_tripped_float{ roundtripped_bfloat };

			ASSERT_EQ(std::fpclassify(round_tripped_float), float_category);
			ASSERT_EQ(std::signbit(round_tripped_float), std::signbit(f));

			if (((i >= _32641) && (i <= _32703)) ||
				((i >= _65409) && (i <= _65471)))
			{
				// i has the raw bits of a signaling NaN. In this case, the round-trip
				// may not be lossless, as a signaling NaN may change into a quiet NaN
				// when returned by the float() conversion operator of bfloat_t
				// (depending on compiler version and compilation flags).
				// The round-trip bfloat16 -> float -> bfloat16 will yield a quiet NaN
				// anyway, as taken care of by the explicit `bfloat16_t::bfloat16_t(float)`.

				ASSERT_EQ(float_category, FP_NAN);
				ASSERT_EQ(get_raw_bits(roundtripped_bfloat), i + _64);
			}
			else
			{
				ASSERT_TRUE((float_category == FP_NAN) || (float_category == FP_NORMAL) || (float_category == FP_INFINITE));

				// For this i, round-trips are lossless.
				EXPECT_EQ(float_to_array_of_bytes(round_tripped_float), float_to_array_of_bytes(f));
				EXPECT_EQ(float_to_raw_bits_of_bfloat16(f), i);
			}
		}
	}
}


GTEST_TEST(bfloat16, ConversionFromIntegerTypesEqualsConversionFromFloat)
{
	// Exhaustive testing for integer types <= 16 bits:
	assert_conversion_from_integer_type_equals_conversion_from_float<std::int8_t>();
	assert_conversion_from_integer_type_equals_conversion_from_float<std::uint8_t>();
	assert_conversion_from_integer_type_equals_conversion_from_float<std::int16_t>();
	assert_conversion_from_integer_type_equals_conversion_from_float<std::uint16_t>();

	// Limited (incomplete) testing for integer types >= 32 bits, to avoid excessive test duration:
	assert_conversion_from_min_and_max_equals_conversion_from_float<std::int32_t>();
	assert_conversion_from_min_and_max_equals_conversion_from_float<std::uint32_t>();
	assert_conversion_from_min_and_max_equals_conversion_from_float<std::int64_t>();
	assert_conversion_from_min_and_max_equals_conversion_from_float<std::uint64_t>();

	// Test zero for signed types >= 32 bits (already tested for unsigned types by
	// assert_conversion_from_min_and_max_equals_conversion_from_float<T>().
	assert_conversion_from_specified_value_equals_conversion_from_float(std::int32_t{ 0 });
	assert_conversion_from_specified_value_equals_conversion_from_float(std::int64_t{ 0 });

	// And then just test integer types >= 32 bits for values from 65535 down to one:
	for (auto i = uint16_max; i > 0; --i)
	{
		assert_conversion_from_specified_value_equals_conversion_from_float(std::int32_t{ i });
		assert_conversion_from_specified_value_equals_conversion_from_float(std::int64_t{ i });
		assert_conversion_from_specified_value_equals_conversion_from_float(-std::int32_t{ i });
		assert_conversion_from_specified_value_equals_conversion_from_float(-std::int64_t{ i });
		assert_conversion_from_specified_value_equals_conversion_from_float(std::uint32_t{ i });
		assert_conversion_from_specified_value_equals_conversion_from_float(std::uint64_t{ i });
	}
}


#ifndef BIOVAULT_BFLOAT16_TEST_NO_CONVERTING_ASSIGNMENT
GTEST_TEST(bfloat16, AssignmentFromIntegerYieldsSameRawBitsAsConstructionFromInteger)
{
	assert_assignment_yields_same_raw_bits_as_construction_from_value(0);

	// Exhaustive testing for integer types <= 16 bits:
	assert_assignment_from_integer_type_yields_same_raw_bits_as_construction_from_value<std::int8_t>();
	assert_assignment_from_integer_type_yields_same_raw_bits_as_construction_from_value<std::uint8_t>();
	assert_assignment_from_integer_type_yields_same_raw_bits_as_construction_from_value<std::int16_t>();
	assert_assignment_from_integer_type_yields_same_raw_bits_as_construction_from_value<std::uint16_t>();

	// Limited (incomplete) testing for integer types >= 32 bits, to avoid excessive test duration:
	assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value<std::int32_t>();
	assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value<std::uint32_t>();
	assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value<std::int64_t>();
	assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value<std::uint64_t>();

	// Test zero for signed types >= 32 bits (already tested for unsigned types by
	// assert_assignment_from_min_and_max_yield_same_raw_bits_as_construction_from_value<T>().
	assert_assignment_yields_same_raw_bits_as_construction_from_value(std::int32_t{ 0 });
	assert_assignment_yields_same_raw_bits_as_construction_from_value(std::int64_t{ 0 });

	// And then just test integer types >= 32 bits for values from 65535 down to one:
	for (auto i = uint16_max; i > 0; --i)
	{
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::int32_t{ i });
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::int64_t{ i });
		assert_assignment_yields_same_raw_bits_as_construction_from_value(-std::int32_t{ i });
		assert_assignment_yields_same_raw_bits_as_construction_from_value(-std::int64_t{ i });
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::uint32_t{ i });
		assert_assignment_yields_same_raw_bits_as_construction_from_value(std::uint64_t{ i });
	}

}

GTEST_TEST(bfloat16, AssignmentFromFloatYieldsSameRawBitsAsConstructionFromFloat)
{
	for (float f{}; f <= 2.0f; f += 0.5)
	{
		assert_assignment_yields_same_raw_bits_as_construction_from_value(f);
		assert_assignment_yields_same_raw_bits_as_construction_from_value(-f);
	}

	for (float f :
	{
		float_limits::min(),
		float_limits::max(),
		float_limits::epsilon(),
		float_limits::quiet_NaN(),
		float_limits::denorm_min(),
		float_limits::infinity()
	})
	{
		assert_assignment_yields_same_raw_bits_as_construction_from_value(f);
		assert_assignment_yields_same_raw_bits_as_construction_from_value(-f);
	}
}
#endif
