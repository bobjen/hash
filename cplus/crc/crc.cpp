#include "crc.h"
#include <stdio.h>
#include <chrono>
#include <cstring>
#if defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1) || defined(_M_X64)
#  include <x86intrin.h>
#  define HAS_INTRINSICS
#endif

#ifndef nullptr
# define nullptr NULL
#endif

Crc* g_crc = new Crc();

static const u16 c_nByteValues = 256;
static const u16 c_nSliceBits = 8;
static const u16 c_nSliceValues = 1 << c_nSliceBits;


// This produces the same bits as crc64nvme.
// I made an effort for Compute() of small values and Concat() to be fast.
// This polynomial shifts right every bit, not left, so the polynomial gets XORed if (val & 0x1).
// Multiplication is mostly symmetric, but you do carries of low 8 bytes to top, not high 8 bytes to bottom.
// CRC also XORs ~((u64)0) before and after every Compute(), but for Concat(), those all cancel out.
// Define CrcInternal here so the .h file does not need to include the intrinsic .h files.
class CrcInternal
{
public:
    static const u64 c_complement = ~u64(0);
    static const u64 c_poly = 0x9a6c9329ac4bc9b5UL;  // primitive polynomial, other than the 1
    static const int c_nCarries = sizeof(u64);
    static const int c_nBits = 8*sizeof(u64);
    static const int c_folds = 32;
    static const int c_nSlices = 1 + ((c_nBits-1)/c_nSliceBits);
    static const u64 c_msb = (((u64)1) << (c_nBits - 1));  // most significant bit
    u64 *m_carry[c_nCarries];  // 64-bit multiplication yields a 128 bit result: carries for low 8 bytes
    u64 *m_slice[c_nSlices];  // premultiply m_pow[] values for 16-bit slices of "size"
    u16 *m_mult[c_nByteValues];  // m_mult[i][j] is carryless i*j, with i and j both 8-bit values

#ifdef HAS_INTRINSICS
    __m128i m_tab[1 << 4]; // ((4-bit-value *)&m_tab[i])[j] is carryless i*j, with i and j both 4-bit values
    __m128i m_fold[c_folds];  // coefficients for the bulk of the data for compute
    __m128i m_tail;  // coefficients for the last few bytes for compute
    __m128i m_barrett;  // coefficients for Barrett reduction
#endif // HAS_INTRINSICS


#ifdef HAS_INTRINSICS
    __m128i Set128(u64 high, u64 low) const
    {
        __m128i z;
        ((u64 *)&z)[1] = high;
        ((u64 *)&z)[0] = low;
        return z;
    }
#endif // HAS_INTRINSICS
    
    CrcInternal()
    {
#ifdef HAS_INTRINSICS
        // ((u8 *)&m_tab[iFrag])[iValue] is carryless iFrag * iValue
        // _mm_shuffle_epi8 can use this to accomplish 16 4x4-bit multiplications at once
        for (u8 iFrag = 0; iFrag < 16; iFrag++)
        {
            m_tab[iFrag] = _mm_set_epi32(0,0,0,0);
            for (u8 iBit = 0;  iBit < 4;  iBit++)
            {
                if (iFrag & (1 << iBit))
                {
                    for (u8 iValue = 0;  iValue < 16;  iValue++)
                    {
                        ((u8 *)&m_tab[iFrag])[iValue] ^= (iValue << iBit);
                    }
                }
            }
        }

        // initializing multiplication tables is 0.01 seconds, don't delay process start if you do not have to        
        memset(m_mult, 0, sizeof(m_mult));        
#else
        InitMult();
#endif // HAS_INTRINSICS

        // These other tables are always required, and faster to derive on the fly than read off disk
        for (int iCarry = c_nCarries;  iCarry--;)
        {
            u64* table = new u64[c_nByteValues];  // one for each byte value
            if (table == nullptr)
            {
                fprintf(stderr, "error: crc could not allocate m_carry[%d]\n", iCarry);
                exit(1);
            }
            for (int iValue = 0;  iValue < c_nByteValues;  iValue++)
            {
                table[iValue] = 0;
            }
            m_carry[iCarry] = table;
        }

        // carries for multiplying an 8-byte value by c_poly
        u64 pow = c_poly;  // pow is 2^(c_nBits+iBit+8*iCarry) mod c_poly
        for (int iCarry = c_nCarries;  iCarry--;)  // for each carry byte
        {
            u64* table = m_carry[iCarry];
            for (int iBit = 8;  iBit--;)
            {
                for (int iValue = 0;  iValue < c_nByteValues;  iValue++)
                {
                    if (iValue & (1<<iBit))
                    {
                        table[iValue] ^= pow;
                    }
                }
                if (pow & 1)
                {
                    pow = (pow >> 1) ^ c_poly;
                }
                else
                {
                    pow = (pow >> 1);
                }
            }
        }

        u64 m_pow[c_nBits];  // 2^^(2^^i) mod c_poly
        m_pow[0] = ((u64)1) << (c_nBits - 1 - 8); // -8 because length is in bytes not bits
        for (int iPow = 1;  iPow < c_nBits;  iPow++)
        {
            m_pow[iPow] = MulCarryless(m_pow[iPow-1], m_pow[iPow-1]);
        }

        for (u16 iSlice = 0;  iSlice < c_nBits;  iSlice += c_nSliceBits)
        {
            u16 index = iSlice / c_nSliceBits;
            u64* table = new u64[c_nSliceValues];
            if (table == nullptr)
            {
                fprintf(stderr, "error: crc could not allocate m_slice[%d]\n", iSlice);
                exit(1);
            }
            for (u64 iValue = 0;  iValue < c_nSliceValues;  iValue++)
            {
                table[iValue] = c_msb;
                for (u64 iBit = 0;  iBit < c_nSliceBits && iSlice + iBit < c_nBits;  iBit++)
                {
                    if (iValue & (((u64)1) << iBit))
                    {
                        table[iValue] = MulCarryless(table[iValue], m_pow[iSlice + iBit]);
                    }
                }
            }
            m_slice[index] = table;
        }

#ifdef HAS_INTRINSICS        
        for (u64 iFold = 0;  iFold < c_folds;  iFold++)
        {
            m_fold[iFold] =  _mm_set_epi64x(
                MulCarryless(1, Pow256(c_msb, iFold * 16 + 8)),
                MulCarryless(1, Pow256(c_msb, iFold * 16 + 16)));
        }

        // Roll forward 16 bytes, roll back 16 bytes.  I don't know why but these are the values that work.
        m_tail = _mm_set_epi64x(
            MulCarryless(Pow256(c_msb, u64(-1)/8 - 8), c_msb>>6), // roll back (m_tail[1])
            MulCarryless(1, Pow256(c_msb, 8)));  // roll forward (m_tail[0])
        m_barrett = _mm_set_epi32(0, 0, 0, 0);
        ((u64 *)&m_barrett)[0] = 0x27ecfa329aef9f77UL;  // ((2^^128 / c_poly) << 1) | 1
        ((u64 *)&m_barrett)[1] = 0x34d926535897936aUL;  // c_msb/2
#endif // HAS_INTRINSICS
    }


    ~CrcInternal()
    {
        for (u16 x = 0;  x < c_nByteValues;  x++)  // for each first byte
        {
            delete[] m_mult[x];
            m_mult[x] = nullptr;
        }

        for (int iCarry = c_nCarries;  iCarry--;)
        {
            delete[] m_carry[iCarry];
            m_carry[iCarry] = nullptr;
        }

        for (u64 iSlice = 0;  iSlice < c_nSlices;  iSlice++)
        {
            delete[] m_slice[iSlice];
            m_slice[iSlice] = nullptr;
        }
    }


    void InitMult()
    {
        for (u16 x = 0;  x < c_nByteValues;  x++)  // for each first byte
        {
            u16* table = new u16[c_nByteValues];
            if (table == nullptr)
            {
                fprintf(stderr, "could not allocate m_mult[%u]", x);
                exit(1);
            }
            for (u16 y = 0;  y < c_nByteValues;  y++)  // for each second byte
            {
                u16 z = 0;
                for (int iBit = 0;  iBit < 8;  iBit++)
                {
                    if (x & (1 << iBit))
                    {
                        z ^= (y << iBit);
                    }
                }
                table[y] = z;
            }
            m_mult[x] = table;
        }
    }


    // original really slow multiplication definition
    u64 MulPoly (u64 a, u64 b) const
    {
        u64 r;
        u64 twiddle = c_complement;
        u64 msb = twiddle ^ (twiddle>>1);

        for (r = 0;  a != 0;  a <<= 1)
        {
            if (a & msb)
            {
                r ^= b;
                a ^= msb;
            }

            b = (b >> 1 ) ^ ((b & 1) ? c_poly : 0);
        }

        return (r);
    }


    // multiplication, no fancy instructions
    u64 MulKaratsuba(u64 x, u64 y) const
    {
        // Use Karatsuba multiplication, table lookup
        // CRC does bits in backward order, so shift result left by 1 so 0x80*0x80 = 0x8000 not 0x4000
        // *a are 1-byte values
        // *b are 2-byte products of 1-byte values
        // *c are 4-byte products of 2-byte values
        // *d are 8-byte products of 4-byte values
        u64 x0a = x&0xff;
        u64 x1a = (x>>8)&0xff;
        u64 y0a = y&0xff;
        u64 y1a = (y>>8)&0xff;
        u64 t00b = m_mult[x0a][y0a];
        u64 t11b = m_mult[x1a][y1a];
        u64 t01b = m_mult[x0a ^ x1a][y0a ^ y1a] ^ t00b ^ t11b;

        u64 x2a = (x>>16)&0xff;
        u64 x3a = (x>>24)&0xff;
        u64 y2a = (y>>16)&0xff;
        u64 y3a = (y>>24)&0xff;
        u64 t22b = m_mult[x2a][y2a];
        u64 t33b = m_mult[x3a][y3a];
        u64 t23b = m_mult[x2a ^ x3a][y2a ^ y3a] ^ t22b ^ t33b;

        u64 x02a = x0a ^ x2a;
        u64 x13a = x1a ^ x3a;
        u64 y02a = y0a ^ y2a;
        u64 y13a = y1a ^ y3a;
        u64 t02b = m_mult[x02a][y02a];
        u64 t13b = m_mult[x13a][y13a];
        u64 t0123b = m_mult[x02a ^ x13a][y02a ^ y13a] ^ t02b ^ t13b;

        u64 t00c = t00b ^ (t01b << 8) ^ (t11b << 16);
        u64 t22c = t22b ^ (t23b << 8) ^ (t33b << 16);
        u64 t02c = (t02b ^ (t0123b << 8) ^ (t13b << 16)) ^ t00c ^ t22c;

        u64 t00d = t00c ^ (t02c << 16) ^ (t22c << 32);


        u64 x4a = (x>>32)&0xff;
        u64 x5a = (x>>40)&0xff;
        u64 y4a = (y>>32)&0xff;
        u64 y5a = (y>>40)&0xff;
        u64 t44b = m_mult[x4a][y4a];
        u64 t55b = m_mult[x5a][y5a];
        u64 t45b = m_mult[x4a ^ x5a][y4a ^ y5a] ^ t44b ^ t55b;

        u64 x6a = (x>>48)&0xff;
        u64 x7a = (x>>56);
        u64 y6a = (y>>48)&0xff;
        u64 y7a = (y>>56);
        u64 t66b = m_mult[x6a][y6a];
        u64 t77b = m_mult[x7a][y7a];
        u64 t67b = m_mult[x6a ^ x7a][y6a ^ y7a] ^ t66b ^ t77b;

        u64 x46a = x4a ^ x6a;
        u64 x57a = x5a ^ x7a;
        u64 y46a = y4a ^ y6a;
        u64 y57a = y5a ^ y7a;
        u64 t46b = m_mult[x46a][y46a];
        u64 t57b = m_mult[x57a][y57a];
        u64 t4567b = m_mult[x46a ^ x57a][y46a ^ y57a] ^ t46b ^ t57b;

        u64 t44c = t44b ^ (t45b << 8) ^ (t55b << 16);
        u64 t66c = t66b ^ (t67b << 8) ^ (t77b << 16);
        u64 t46c = (t46b ^ (t4567b << 8) ^ (t57b << 16)) ^ t44c ^ t66c;

        u64 t44d = t44c ^ (t46c << 16) ^ (t66c << 32);


        u64 x04a = x0a ^ x4a;
        u64 x15a = x1a ^ x5a;
        u64 y04a = y0a ^ y4a;
        u64 y15a = y1a ^ y5a;
        u64 t04b = m_mult[x04a][y04a];
        u64 t15b = m_mult[x15a][y15a];
        u64 t0145b = m_mult[x04a ^ x15a][y04a ^ y15a] ^ t04b ^ t15b;

        u64 x26a = x2a ^ x6a;
        u64 x37a = x3a ^ x7a;
        u64 y26a = y2a ^ y6a;
        u64 y37a = y3a ^ y7a;
        u64 t26b = m_mult[x26a][y26a];
        u64 t37b = m_mult[x37a][y37a];
        u64 t2367b = m_mult[x26a ^ x37a][y26a ^ y37a] ^ t26b ^ t37b;

        u64 x0246a = x04a ^ x26a;
        u64 x1357a = x15a ^ x37a;
        u64 y0246a = y04a ^ y26a;
        u64 y1357a = y15a ^ y37a;
        u64 t0246b = m_mult[x0246a][y0246a];
        u64 t1357b = m_mult[x1357a][y1357a];
        u64 t01234567b = m_mult[x0246a ^ x1357a][y0246a ^ y1357a] ^ t0246b ^ t1357b;

        u64 t04c = t04b ^ (t0145b << 8) ^ (t15b << 16);
        u64 t26c = t26b ^ (t2367b << 8) ^ (t37b << 16);
        u64 t0246c = (t0246b ^ (t01234567b << 8) ^ (t1357b << 16)) ^ t04c ^ t26c;

        u64 t04d = (t04c ^ (t0246c << 16) ^ (t26c << 32)) ^ t00d ^ t44d;


        // low 64-bits and high 64-bits, before carries
        // CRC multiply does bits in backwards order, so 0x80*0x80 should be 0x8000 not 0x4000, so shift left by 1
        u64 rLo = (t00d << 1) ^ (t04d << 33);
        u64 rHi = (t04d >> 31) ^ (t44d << 1);

        return
            rHi ^
            m_carry[7][rLo>>56] ^
            m_carry[6][(rLo>>48)&0xff] ^
            m_carry[5][(rLo>>40)&0xff] ^
            m_carry[4][(rLo>>32)&0xff] ^
            m_carry[3][(rLo>>24)&0xff] ^
            m_carry[2][(rLo>>16)&0xff] ^
            m_carry[1][(rLo>>8)&0xff] ^
            m_carry[0][rLo&0xff];
    }


    u64 MulCarryless(u64 a, u64 b) const
    {
#ifndef HAS_INTRINSICS
        return MulKaratsuba(a, b);
#else
        // calculate the full z=x*y, not mod anything
        __m128i xx = Set128(0, a);
        __m128i yy = Set128(0, b);
        __m128i zz = _mm_clmulepi64_si128(xx,yy,0x00);

        u64 rLo0 = ((u64 *)&zz)[0];
        u64 rHi0 = ((u64 *)&zz)[1];
        u64 rLo = rLo0 << 1;
        u64 rHi = (rLo0 >> 63) ^ (rHi0 << 1);

        return
            rHi ^
            m_carry[7][rLo>>56] ^
            m_carry[6][(rLo>>48)&0xff] ^
            m_carry[5][(rLo>>40)&0xff] ^
            m_carry[4][(rLo>>32)&0xff] ^
            m_carry[3][(rLo>>24)&0xff] ^
            m_carry[2][(rLo>>16)&0xff] ^
            m_carry[1][(rLo>>8)&0xff] ^
            m_carry[0][rLo&0xff];
#endif // HAS_INTRINSICS
    }

    // compute the CRC a byte at a time using MulPoly()
    u64 ComputePoly(
        const void* pSrc,
        u64 uSize,
        u64 uCrc) const
    {
        u64 crc = ~uCrc;
        for (u64 i = 0;  i < uSize;  i++)
        {
            crc ^= u64(((u8 *)pSrc)[i]);
            crc = MulPoly(crc, 0x80000000000000);
        }
        return ~crc;
    }


    // multiply A by (256 to the Bth power)
    u64 Pow256(u64 a, u64 b) const
    {
        for (int iSlice = 0;  b > 0;  iSlice++)
        {
            a = MulCarryless(a, m_slice[iSlice][b & (c_nSliceValues - 1)]);
            b >>= c_nSliceBits;
        }
        return a;
    }

#ifdef HAS_INTRINSICS

#define STEP(index, offset, sum) \
    fold = m_fold[index]; \
    y = _mm_clmulepi64_si128(z, fold, 0x11); \
    t = _mm_clmulepi64_si128(z, fold, 0x00); \
    z = _mm_loadu_si128((const __m128i*)(pData+(offset))); \
    t = _mm_xor_si128(t, y); \
    sum = _mm_xor_si128(sum, t)

    // ClMul is carryless multiply (XOR instead of +), 8x8=16 bytes.  You handle 8 bytes of input by
    // XORing it to the current CRC and multiplying by c_poly (the m_carry[] tables accomplish that,
    // or you can use ClMul).  Folding a 16-byte state into an 8-byte result can be done using the
    // m_carry tables or "Barrett reduction".
    //
    // Rather than folding 16-byte results into 8-byte CRCs after each ClMul, we keep a 16-byte state.
    // Further inputs are XORed 16 bytes at a time, and the high and low 8 bytes use ClMul with coefficients
    // that differ by c_poly so both 8x8=16-byte results are pulled ahead the correct amount and can be
    // XORed together to produce a new 16-byte state.  The 16 byte state is only folded into an 8-byte
    // CRC once at the very end.  x and xx are parallel 16-byte states of different parts of the input.
    //
    // CRC is bit-mirrored, meaning c_msb is unity, c_msb>>1 is unity times 2, and 1 is 2^^63.
    // ClMul assumes high bits are high, and is right-aligned, but CRC treats low bits as high and
    // is left-aligned, so any ClMul results has been effectively multiplied by an extra c_msb>>1,
    // so the coefficients given to ClMul are effectively divided by 2 to compensate for that.
    //
    // I am not sure if the bottleneck here is memory bandwith or ClMul.  Aligned vs unaligned reads
    // (load vs loadu) made no difference.  ClMul is slow, but many can go in parallel, how many in
    // parallel and how slow varies by platform.  With infinite parallelism the XOR chain of x, xx
    // is the bottleneck, but that isn't the actual bottleneck on any platform so far.
    //
    // If-statements are slow if their condition isn't known until late.  Switch-statements are about
    // the cost of a mispredicted if-statement regardless of the number of cases.  While-loops are a
    // lot of if-statements, so replacing a while-loop with a switch with fallthrough can be a win.
    // Table lookup is often faster than if-statements, and table lookups can go in parallel, so replacing
    // switch statements or if-statements with table lookup is often a win.  But I tried replacing the
    // tail16 switch statement with table-lookup (and shuffle instead of srli) and it was slower.
    //
    // Compute() is about 10x times slower per byte for 2-byte inputs than 2-million-byte inputs, and
    // calling Compute() for very small values is unfortunately very common, so this makes an effort
    // to make very small keys as fast as possible.  For very big keys only the while-loop matters.
    u64 Compute(
        const void* pSrc,
        u64 uSize,
        u64 uCrc) const
    {
        u8* pData = (u8 *)pSrc;

        uCrc = ~uCrc;

        switch (uSize)
        {
        default:
        {
            __m128i x;    // state
            __m128i xx;   // more state: unclear if it helps, but does not hurt
            __m128i t;    // temp
            __m128i y;    // more temp
            __m128i z;    // holds data that has been read
            __m128i fold; // coefficients for multiplies

            // read in the first 16 bytes
            z = _mm_loadu_si128((const __m128i*)pData);
            z = _mm_xor_si128(_mm_set_epi64x(0, uCrc), z);
            pData += 16;

            // in/out: state in z, next 16 bytes already in z not yet folded.
            // Is 256 better than 128 or 512?  Unclear.
            while (uSize > 256+16)
            {
                fold = m_fold[15];
                y = _mm_clmulepi64_si128(z, fold, 0x11);
                t = _mm_clmulepi64_si128(z, fold, 0x00);
                z = _mm_loadu_si128((const __m128i*)pData);
                x = _mm_xor_si128(t, y); // x is assigned not summed

                fold = m_fold[14];
                y = _mm_clmulepi64_si128(z, fold, 0x11);
                t = _mm_clmulepi64_si128(z, fold, 0x00);
                z = _mm_loadu_si128((const __m128i*)(pData+16));
                xx = _mm_xor_si128(t, y); // xx is assigned not summed

                STEP(13, 2*16, x);
                STEP(12, 3*16, xx);
                STEP(11, 4*16, x);
                STEP(10, 5*16, xx);
                STEP(9,  6*16, x);
                STEP(8,  7*16, xx);
                STEP(7,  8*16, x);
                STEP(6,  9*16, xx);
                STEP(5, 10*16, x);
                STEP(4, 11*16, xx);
                STEP(3, 12*16, x);
                STEP(2, 13*16, xx);
                STEP(1, 14*16, x);
                STEP(0, 15*16, xx);

                z = _mm_xor_si128(x, z);
                z = _mm_xor_si128(xx, z);

                pData += 256;
                uSize -= 256;
            }

            // In: State and next 16 bytes in z.  uSize includes next 16 bytes.
            // Out: state in z, with last 8 bytes in z0.
            x = _mm_set_epi32(0,0,0,0);
            xx = _mm_set_epi32(0,0,0,0);
            u64 tail16 = (uSize-1) & ~u64(0xf); // all but the last 1..16 bytes
            pData += tail16-16;
            uSize -= tail16;
            switch (tail16/16)  // cases purposely fall through
            {
            default:
                fprintf(stderr, "error: crc, missing switch\n");
                exit(1);

            case 16:
                fold = m_fold[15];
                y = _mm_clmulepi64_si128(z, fold, 0x11);
                t = _mm_clmulepi64_si128(z, fold, 0x00);
                z = _mm_loadu_si128((const __m128i*)(pData-15*16));
                x = _mm_xor_si128(t, y); // x is assigned not summed
                // fall through

            case 15:
                fold = m_fold[14];
                y = _mm_clmulepi64_si128(z, fold, 0x11);
                t = _mm_clmulepi64_si128(z, fold, 0x00);
                z = _mm_loadu_si128((const __m128i*)(pData-14*16));
                xx = _mm_xor_si128(t, y); // xx is assigned not summed
                // fall through

            case 14:
                STEP(13, -13*16, x); // fall through
            case 13:
                STEP(12, -12*16, xx); // fall through
            case 12:
                STEP(11, -11*16, x); // fall through
            case 11:
                STEP(10, -10*16, xx); // fall through
            case 10:
                STEP(9, -9*16, x); // fall through
            case 9:
                STEP(8, -8*16, xx); // fall through
            case 8:
                STEP(7, -7*16, x); // fall through
            case 7:
                STEP(6, -6*16, xx); // fall through
            case 6:
                STEP(5, -5*16, x); // fall through
            case 5:
                STEP(4, -4*16, xx); // fall through
            case 4:
                STEP(3, -3*16, x); // fall through
            case 3:
                STEP(2, -2*16, xx); // fall through
            case 2:
                STEP(1, -16, x); // fall through
            case 1:
                // state in x and xx, with 16+1..16 more bytes, first 16 bytes in z
                fold = m_fold[0];
                y = _mm_clmulepi64_si128(z, fold, 0x11);
                t = _mm_clmulepi64_si128(z, fold, 0x00);
                t = _mm_xor_si128(t, y);
                xx = _mm_xor_si128(xx, t);

                // stuff all the state into x
                x = _mm_xor_si128(x, xx);

                // Read the last 1..16 bytes into z
                // Note this overlaps previous reads so you have to shift out bytes that are already read.
                z = _mm_loadu_si128((const __m128i*)(pData+uSize-16));
                fold = m_tail;

                // Handle the last 1..16 bytes.  x is full, z is partially full and left-aligned.
                switch (uSize)
                {
                case 16:
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 8);
                    x = _mm_clmulepi64_si128(x, fold, 0x00);
                    break;
                case 15:
                    z = _mm_srli_si128(z, 1);  // srli takes an immediate, made possible by the "switch"
                    x = _mm_xor_si128(x, z);   // xor the right-aligned z into x
                    z = _mm_srli_si128(x, 7);  // z[0] holds the first 8 bytes of input
                    t = _mm_slli_si128(x, 1);  // t[0] now has last 7 bits of input and a low byte of 0
                    x = _mm_clmulepi64_si128(t, fold, 0x00);  // fold t[0] so it lines up with z[0]
                    break;
                case 14:
                    z = _mm_srli_si128(z, 2);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 6);
                    t = _mm_slli_si128(x, 2);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 13:
                    z = _mm_srli_si128(z, 3);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 5);
                    t = _mm_slli_si128(x, 3);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 12:
                    z = _mm_srli_si128(z, 4);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 4);
                    t = _mm_slli_si128(x, 4);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 11:
                    z = _mm_srli_si128(z, 5);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 3);
                    t = _mm_slli_si128(x, 5);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 10:
                    z = _mm_srli_si128(z, 6);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 2);
                    t = _mm_slli_si128(x, 6);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 9:
                    z = _mm_srli_si128(z, 7);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 1);
                    t = _mm_slli_si128(x, 7);
                    x = _mm_clmulepi64_si128(t, fold, 0x00);
                    break;
                case 8:
                    z = _mm_srli_si128(z, 8);
                    break;
                case 7:
                    z = _mm_srli_si128(z, 9);  // right-align the input
                    x = _mm_xor_si128(x, z);   // add the input to the state
                    z = _mm_srli_si128(x, 7);  // find the piece of the state that is beyond all input
                    z = _mm_clmulepi64_si128(z, fold, 0x11); // it went too far, undo it
                    x = _mm_slli_si128(x, 1);  // align the input so it will fold right
                    break;
                case 6:
                    z = _mm_srli_si128(z, 10);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 6);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 2);
                    break;
                case 5:
                    z = _mm_srli_si128(z, 11);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 5);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 3);
                    break;
                case 4:
                    z = _mm_srli_si128(z, 12);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 4);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 4);
                    break;
                case 3:
                    z = _mm_srli_si128(z, 13);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 3);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 5);
                    break;
                case 2:
                    z = _mm_srli_si128(z, 14);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 2);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 6);
                    break;
                case 1:
                    z = _mm_srli_si128(z, 15);
                    x = _mm_xor_si128(x, z);
                    z = _mm_srli_si128(x, 1);
                    z = _mm_clmulepi64_si128(z, fold, 0x11);
                    x = _mm_slli_si128(x, 7);
                    break;
                default:
                    fprintf(stderr, "error: crc, missing switch value\n");
                    exit(1);
                }
                break;
            }
            x = _mm_xor_si128(x, z);

            // Barrett reduction
            fold = m_barrett;
            z = _mm_clmulepi64_si128(x, fold, 0x00);
            t = _mm_slli_si128(z, 8);
            z = _mm_clmulepi64_si128(z, fold, 0x10);
            x = _mm_xor_si128(x, t);
            z = _mm_xor_si128(z, x);
            uCrc = _mm_extract_epi64(z, 1);
            
            break;
        }

        // otherwise it is a short key, 0..16 bytes
        case 16:
        {
            __m128i z = _mm_loadu_si128((const __m128i*)pData);
            __m128i x = _mm_srli_si128(z, 8);
            z = _mm_xor_si128(_mm_set_epi64x(0, uCrc), z);
            z = _mm_clmulepi64_si128(z, m_fold[0], 0x10);
            x = _mm_xor_si128(x, z);

            // Barrett reduction
            __m128i barrett = m_barrett;
            z = _mm_clmulepi64_si128(x, barrett, 0x00);
            __m128i t = _mm_slli_si128(z, 8);
            z = _mm_clmulepi64_si128(z, barrett, 0x10);
            x = _mm_xor_si128(x, t);
            z = _mm_xor_si128(z, x);
            uCrc = _mm_extract_epi64(z, 1);

            break;
        }
        case 15:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            u64 x = *(u64 *)(&pData[uSize-8]) ^ (uCrc << 8);
            uCrc = (uCrc >> 56) ^ 
                m_carry[1][(x >> 8) & 0xff] ^
                m_carry[2][(x >> 16) & 0xff] ^
                m_carry[3][(x >> 24) & 0xff] ^
                m_carry[4][(x >> 32) & 0xff] ^
                m_carry[5][(x >> 40) & 0xff] ^
                m_carry[6][(x >> 48) & 0xff] ^
                m_carry[7][x >> 56];
            break;
        }
        case 14:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            u64 x = *(u64 *)(&pData[uSize-8]) ^ (uCrc << 16);
            uCrc = (uCrc >> 48) ^ 
                m_carry[2][(x >> 16) & 0xff] ^
                m_carry[3][(x >> 24) & 0xff] ^
                m_carry[4][(x >> 32) & 0xff] ^
                m_carry[5][(x >> 40) & 0xff] ^
                m_carry[6][(x >> 48) & 0xff] ^
                m_carry[7][x >> 56];
            break;
        }
        case 13:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            u64 x = *(u64 *)(&pData[uSize-8]) ^ (uCrc << 24);
            uCrc = (uCrc >> 40) ^ 
                m_carry[3][(x >> 24) & 0xff] ^
                m_carry[4][(x >> 32) & 0xff] ^
                m_carry[5][(x >> 40) & 0xff] ^
                m_carry[6][(x >> 48) & 0xff] ^
                m_carry[7][x >> 56];
            break;
        }
        case 12:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            uCrc ^= *((u32 *)&pData[8]);
            uCrc = (uCrc >> 32) ^
                m_carry[4][uCrc & 0xff] ^
                m_carry[5][(uCrc >> 8) & 0xff] ^
                m_carry[6][(uCrc >> 16) & 0xff] ^
                m_carry[7][(uCrc >> 24) & 0xff];
            break;
        }
        case 11:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            u64 x = *(u64 *)(&pData[uSize-8]) ^ (uCrc << 40);
            uCrc = (uCrc >> 24) ^ 
                m_carry[5][(x >> 40) & 0xff] ^
                m_carry[6][(x >> 48) & 0xff] ^
                m_carry[7][x >> 56];
            break;
        }
        case 10:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            uCrc ^= *((u16 *)&pData[8]);
            uCrc = (uCrc >> 16) ^
                m_carry[6][uCrc & 0xff] ^
                m_carry[7][(uCrc >> 8) & 0xff];
            break;
        }
        case 9:
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
            // fall through
        case 8:
        {
            uCrc ^= *(u64 *)pData;
            uCrc = m_carry[0][uCrc & 0xff] ^
                m_carry[1][(uCrc >> 8) & 0xff] ^
                m_carry[2][(uCrc >> 16) & 0xff] ^
                m_carry[3][(uCrc >> 24) & 0xff] ^
                m_carry[4][(uCrc >> 32) & 0xff] ^
                m_carry[5][(uCrc >> 40) & 0xff] ^
                m_carry[6][(uCrc >> 48) & 0xff] ^
                m_carry[7][uCrc >> 56];
            break;
        }
        case 7:
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
            // fall through
        case 6:
        {
            uCrc ^= *((u32 *)pData);
            uCrc = (uCrc >> 32) ^
                m_carry[4][uCrc & 0xff] ^
                m_carry[5][(uCrc >> 8) & 0xff] ^
                m_carry[6][(uCrc >> 16) & 0xff] ^
                m_carry[7][(uCrc >> 24) & 0xff];
            uCrc ^= *((u16 *)&pData[4]);
            uCrc = (uCrc >> 16) ^
                m_carry[6][uCrc & 0xff] ^
                m_carry[7][(uCrc >> 8) & 0xff];
            break;
        }
        case 5:
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
            // fall through
        case 4:
        {
            uCrc ^= *((u32 *)pData);
            uCrc = (uCrc >> 32) ^
                m_carry[4][uCrc & 0xff] ^
                m_carry[5][(uCrc >> 8) & 0xff] ^
                m_carry[6][(uCrc >> 16) & 0xff] ^
                m_carry[7][(uCrc >> 24) & 0xff];
            break;
        }
        case 3:
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
            // fall through
        case 2:
        {
            uCrc ^= *((u16 *)pData);
            uCrc = (uCrc >> 16) ^
                m_carry[6][uCrc & 0xff] ^
                m_carry[7][(uCrc >> 8) & 0xff];
            break;
        }
        case 1:
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
            // fall through
        case 0:
            // fall through
            ;
        }

        return ~uCrc;
    }

#undef STEP

#else // not HAS_INTRINSICS

    u64 Compute(
        const void* pSrc,
        u64 uSize,
        u64 uCrc) const
    {
        u8* pData = (u8 *)pSrc;
        uCrc = ~uCrc;
        for (u64 i = 0; i < uSize; i++)
            uCrc = (uCrc >> 8) ^ m_carry[7][(uCrc & 0xff) ^ *pData++];
        return ~uCrc;
    }

#endif // HAS_INTRINSICS


    u64 Concat(
        u64 uInitialCrcAB,
        u64 uInitialCrcA,
        u64 uFinalCrcA,
        u64 uSizeA,
        u64 uInitialCrcB,
        u64 uFinalCrcB,
        u64 uSizeB) const
    {
        u64 state = 0;
        // Start the CRC before A.  Start with uInitialCrcAB and cancel out uInitialCrcA.
        if (uInitialCrcAB != uInitialCrcA)
        {
            state = uInitialCrcAB ^ uInitialCrcA;

            // move the CRC state forward by uSizeA
            state = Pow256(state, uSizeA);
        }

        // Now after A and before B.  Account for uFinalCrcA and cancel out uInitialCrcB.
        state ^= uFinalCrcA ^ uInitialCrcB;

        // move the CRC state forward by uSizeB.
        state = Pow256(state, uSizeB);

        // Now after B.  Account for uFinalCrcB.
        state ^= uFinalCrcB;

        return state;
    }


    // Return nullptr if the self-check says the tables are all correct, otherwise a message
    // If a machine has hardware memory corruptions, it could compute concat wrong.  This will detect that.
    const char* MemoryCheck() const
    {
        u64 a = 0;
        const char* rsl = nullptr;
#ifdef HAS_INTRINSICS
        a = Compute(m_tab, sizeof(m_tab), a);
        if (a != 0xf6eb8d32169ecb54)
        {
            printf("tab %lx\n", a);
            rsl = "m_tab corrupt";
        }
#endif

        a = 0;
        for (int i = 0;  i < c_nCarries;  i++)
            a = Compute(m_carry[i], sizeof(u64)*c_nByteValues, a);
        if (a != 0xf5118d5f47061a81)
        {
            printf("carry %lx\n", a);
            rsl = "m_carry corrupt";
        }

        if (m_mult[0] != nullptr)
        {
            a = 0;

            for (int i = 0;  i < c_nByteValues;  i++)
            {
                a = Compute(m_mult[i], sizeof(u16)*c_nByteValues, a);
            }

            if (a != 0x63512d28f5b05fcf)
            {
                printf("mult %lx\n", a);
                rsl = "m_mult corrupt";
            }
        }

        a = 0;
        for (int i = 0;  i < c_nBits/c_nSliceBits;  i++)
        {
            a = Compute(m_slice[i], sizeof(u64)*c_nSliceValues, a);
        }
        if (a != 0x819849c4eb74f7ab)
        {
            printf("slice %lx\n", a);
            rsl = "m_slice corrupt";
        }

        return rsl;
    }

    static u64 Mix(u64 x)
    {
        x ^= ~(x >> 32);
        x *= 0xdeadbeefdeadbeef;
        x ^= x >> 32;
        x *= 0xdeadbeefdeadbeef;
        return x;
    }

    // return true if successful
    bool SelfTest()
    {
        // make sure m_mult is defined
        if (m_mult[0] == nullptr)
        {
            InitMult();
        }

        // do a test of crc64nvme you can verify from the internet
        u64 crc = g_crc->Compute("hello world!", 12, 0);
        u64 expected = 0xd9160d1fa8e418e3;
        if (crc != expected)
        {
            fprintf(stderr, "hello world! got %lx expected %lx\n", crc, expected);
            return false;
        }

        u64 x = 0xc01105a1deadbeef;
        if (MulPoly(crc, x) != MulCarryless(crc, x))
        {
            fprintf(stderr, "MulPoly, MulCarryless inconsistent\n");
            return false;
        }
        if (MulPoly(crc, x) != MulKaratsuba(crc, x))
        {
            fprintf(stderr, "MulPoly, MulKaratsuba inconsistent\n");
            return false;
        }

        // fill a midsized buffer with predictable noise
        static const u64 c_buflen = 1000;
        u64 buf[c_buflen];
        for (u64 i = 0; i < c_buflen; i++)
        {
            buf[i] = x;
            x = Mix(x);
        }

        // test complicated Compute() matches simple ComputePoly()
        for (u64 i = 0; i < c_buflen * sizeof(u64); i++)
        {
            if (ComputePoly(buf, i, 0) != Compute(buf, i, 0))
            {
                fprintf(stderr, "Compute len %lu failed, init 0\n", i);
                return false;
            }
            if (ComputePoly(buf, i, x) != Compute(buf, i, x))
            {
                fprintf(stderr, "Compute len %lu failed, init %lx\n", i, x);
                return false;
            }
        }

        u64 zeroCrc = Compute(buf, 0, 0);
        if (zeroCrc != Concat(0, 0, zeroCrc, 0, 0, zeroCrc, 0))
        {
            fprintf(stderr, "Concat of nulls failed\n");
            return false;
        }

        u64 bufcrc = Compute(buf, c_buflen*sizeof(u64), 0);
        for (u64 i = 0;  i < c_buflen * sizeof(u64); i++)
        {
            u64 a = Compute(buf, i, 0);
            u64 b = Compute(((u8 *)buf)+i, c_buflen*sizeof(u64)-i, 0);
            if (Concat(0, 0, a, i, 0, b, c_buflen*sizeof(u64)-i) != bufcrc)
            {
                fprintf(stderr, "Concat does not match Compute\n");
                return false;
            }
        }

        for (u64 i = 0;  i < 100;  i++)
        {
            // a b c are 8-byte integers, a <= b <= c
            u64 a = x;  x = Mix(x);
            u64 b = x;  x = Mix(x);
            u64 c = x;  x = Mix(x);
            if (a > b) { u64 temp = a; a = b; b = temp; }
            if (a > c) { u64 temp = a; a = c; c = temp; }
            if (b > c) { u64 temp = b; b = c; c = temp; }

            // a+d+e = c
            u64 d = b-a;
            u64 e = c-b;
            if (a+d+e != c) printf("hi bob\n");

            // starting and ending CRCs for a, d, e, ad, de
            u64 a0 = x;  x = Mix(x);
            u64 a1 = x;  x = Mix(x);
            u64 d0 = x;  x = Mix(x);
            u64 d1 = x;  x = Mix(x);
            u64 e0 = x;  x = Mix(x);
            u64 e1 = x;  x = Mix(x);
            u64 ad0 = x;  x = Mix(x);
            u64 ad1 = Concat(ad0, a0, a1, a, d0, d1, d);
            u64 de0 = x;  x = Mix(x);
            u64 de1 = Concat(de0, d0, d1, d, e0, e1, e);

            // starting CRC for a+d+e=c
            u64 c0 = x;  x = Mix(x);
            
            if (Concat(c0, a0, a1, a, de0, de1, d+e) !=
                Concat(c0, ad0, ad1, a+d, e0, e1, e))
            {
                fprintf(stderr, "Concat failed\n");
                return false;
            }
        }
        
        return true;
    }

    void TimingTest()
    {
        // make sure m_mult is defined
        if (m_mult[0] == nullptr)
        {
            InitMult();
        }

        u64 iter = 10000000;
        u64 x = 0xdeadbeefdeadbeef;
        auto a = std::chrono::high_resolution_clock::now();
        for (u64 i = 0; i < iter; i++)
            x += MulCarryless(x, 0xdeadbeefdeadbeef); 
        auto z = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = z - a;
        fprintf(stderr, "MulCarryless() (SIMD instructions) took %g microseconds %lx\n",
                d.count()*1000000/iter, x);

        iter = 10000000;
        x = 0xdeadbeefdeadbeef;
        a = std::chrono::high_resolution_clock::now();
        for (u64 i = 0; i < iter; i++)
            x += MulKaratsuba(x, 0xdeadbeefdeadbeef); 
        z = std::chrono::high_resolution_clock::now();
        d = z - a;
        fprintf(stderr, "MulKaratsuba() (no SIMD) took %g microseconds %lx\n",
                d.count()*1000000/iter, x);
        
        const u64 c_bufLen = 1<<16;
        char buf[c_bufLen];
        for (u64 i = 0; i < c_bufLen; i++)
            buf[i] = (char)(i + d.count());

        iter = 10000;
        x = 0;
        a = std::chrono::high_resolution_clock::now();
        for (u64 i = 0; i < iter; i++)
            x += ComputePoly(buf, 256, x); 
        z = std::chrono::high_resolution_clock::now();
        d = z - a;
        fprintf(stderr, "ComputePoly() (slow) of 256 byte buffer took %g microseconds %lx\n",
                d.count()*1000000/iter, x);

        iter = 10000;
        x = 0;
        a = std::chrono::high_resolution_clock::now();
        for (u64 i = 0; i < iter; i++)
            x += Compute(buf, 256, x); 
        z = std::chrono::high_resolution_clock::now();
        d = z - a;
        fprintf(stderr, "Compute() of 256 byte buffer took %g microseconds %lx\n",
                d.count()*1000000/iter, x);

        a = std::chrono::high_resolution_clock::now();
        x = 0;
        iter = 10000000;
        for (u64 i = 0; i < iter; i++)
            x += Concat(i, i, x, i, i, x, i);
        z = std::chrono::high_resolution_clock::now();
        d = z - a;
        fprintf(stderr, "Concat() took %g microseconds %lu\n\n",
                d.count()*1000000/iter, x);

        for (u64 len = 0; len < 32; len++)
        {
            iter = 1000000;
            x = 0;
            a = std::chrono::high_resolution_clock::now();
            for (u64 i = 0; i < iter; i++)
                x += Compute(buf, len, x); 
            z = std::chrono::high_resolution_clock::now();
            d = z - a;
            fprintf(stderr, "Compute() of %lu byte buffer took %g microseconds %lx\n",
                    len, d.count()*1000000/iter, x);
        }

        for (u64 len = 1; len <= 1<<16; len *= 2)
        {
            iter = 1000000/len;
            x = 0;
            a = std::chrono::high_resolution_clock::now();
            for (u64 i = 0; i < iter; i++)
                x += Compute(buf, len, x); 
            z = std::chrono::high_resolution_clock::now();
            d = z - a;
            fprintf(stderr, "Compute() of %lu byte buffer took %g microseconds %lx\n",
                    len, d.count()*1000000/iter, x);
        }
    }
};


Crc::Crc()
{
    m_internal = new CrcInternal();
    if (m_internal == nullptr)
    {
        fprintf(stderr, "error: crc, could not init internal\n");
        exit(1);
    }
#ifndef HAS_INTRINSICS
    m_internal->InitMult();
#endif
    const char* errMessage = MemoryCheck();
    if (errMessage != nullptr)
    {
        fprintf(stderr, "crc: internal tables wrong, %s\n", errMessage);
        exit(1);
    }
}

Crc::~Crc()
{
    delete m_internal;
}

u64 Crc::Compute(
    const void* pSrc,
    u64 uSize,
    u64 uCrc) const
{
    return m_internal->Compute(pSrc, uSize, uCrc);
}

u64 Crc::Concat(u64 iab, u64 ia, u64 fa, u64 sa, u64 ib, u64 fb, u64 sb) const
{
    return m_internal->Concat(iab, ia, fa, sa, ib, fb, sb);
}

const char* Crc::MemoryCheck() const
{
    return m_internal->MemoryCheck();
}

bool Crc::SelfTest() const
{
    return m_internal->SelfTest();
}

void Crc::TimingTest() const
{
    m_internal->TimingTest();
}
