// 64-bit CRCs (Cyclic Redundancy Checks)
// For noncryptographic detection of corruptions and bitflips

#include <stddef.h>

#ifdef _MSC_VER
  typedef  unsigned __int64 u8;
  typedef  unsigned __int32 u4;
  typedef  unsigned __int16 u2;
  typedef  unsigned __int8  u1;
#else
# include <stdint.h>
  typedef  uint64_t  u8;
  typedef  uint32_t  u4;
  typedef  uint16_t  u2;
  typedef  uint8_t   u1;
#endif

class CrcInternal;
class CrcTest;

class Crc
{
    // disable copying and assigning this class
    Crc(const Crc&) = delete;
    Crc& operator=(const Crc&) = delete;

    CrcInternal* m_internal;

public:
    Crc();
    ~Crc();

    // compute the CRC of an array src of length len with initial crc crc
    u8 Compute(
        const void* src,
        u8 len,
        u8 crc) const;

    // Given the CRC and size of A and B, compute the CRC of (A concat B).
    u8 Concat(
        u8 initCrcAB,
        u8 initCrcA,
        u8 finalCrcA,
        u8 sizeA,
        u8 initCrcB,
        u8 finalCrcB,
        u8 sizeB) const;

    // Corrupt CRCs may be due to corrupt internal tables.
    // Check internal tables are not corrupt (using CRC and known values).
    // Return nullptr if the tables are OK.
    const char* MemoryCheck() const;

    bool SelfTest() const;
};

extern Crc* g_crc;
