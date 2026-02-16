// 64-bit CRCs (Cyclic Redundancy Checks)
// For noncryptographic detection of corruptions and bitflips

#include <stddef.h>

#ifdef _MSC_VER
  typedef  unsigned __int64 u64;
  typedef  unsigned __int32 u32;
  typedef  unsigned __int16 u16;
  typedef  unsigned __int8  u8;
#else
# include <stdint.h>
  typedef  uint64_t  u64;
  typedef  uint32_t  u32;
  typedef  uint16_t  u16;
  typedef  uint8_t   u8;
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
    u64 Compute(
        const void* src,
        u64 len,
        u64 crc) const;

    // Given the CRC and size of A and B, compute the CRC of (A concat B).
    u64 Concat(
        u64 initCrcAB,
        u64 initCrcA,
        u64 finalCrcA,
        u64 sizeA,
        u64 initCrcB,
        u64 finalCrcB,
        u64 sizeB) const;

    // Corrupt CRCs may be due to corrupt internal tables.
    // Check internal tables are not corrupt (using CRC and known values).
    // Return nullptr if the tables are OK.
    const char* MemoryCheck() const;

    bool SelfTest() const;
    void TimingTest() const;
};

extern Crc* g_crc;
