
static inline long floor_div(long x,long y){
    long r = x / y;
    if ((x ^ y) < 0 && (r * y != x)) {
        r-=1;
    }
    return r;
}


inline int random_next (long *random, int bits) {
    *random = (*random * 0x5DEECE66DUL + 0xBUL) & ( (1UL << 48) - 1);
    return (int)(*random >> (48 - bits));
}

inline long random_next_long (long *random) {
  return (((long)random_next(random, 32)) << 32) + random_next(random, 32);
}


__kernel void start(ulong offset, ulong stride, __global ulong *seeds,__global ulong *debug, __global uint *ret) {
	size_t id = get_global_id(0);
	uint max_count = 0;
	ushort max_count_1 = 0;
	ushort max_count_2 = 0;
	ushort max_count_3 = 0;
	ushort max_count_4 = 0;
	ulong seed_base = (offset + id) * stride;
	for (ulong i = 0; i < stride; i++) {
		long world_seed = seed_base|i;
		long r =(world_seed ^ 0x5DEECE66DUL) & ((1UL << 48) - 1);
		long next_long = random_next_long(&r);
        long original=world_seed ^ 0x5DEECE66DUL;
        seeds[id] = world_seed;

        next_long=next_long & 0xFFFFFFFFFFFFUL;
        long lower_bits = next_long & 0xffffffffUL;
        long upper_bits=next_long >> 32;
        if ((lower_bits & 0x80000000UL) != 0) {
            upper_bits += 1;
        }
        long bits_of_danger = 1;
        long low_min = lower_bits << 16 - bits_of_danger;
        long low_max = ((lower_bits + 1) << 16 - bits_of_danger) - 1;
        long upper_min = ((upper_bits << 16) - 107048004364969L) >> bits_of_danger;
        long m1lv = floor_div(low_max * -33441 + upper_min * 17549, 1L << 31 - bits_of_danger) + 1;
        long m2lv = floor_div(low_min * 46603 + upper_min * 39761, 1L << 32 - bits_of_danger) + 1;
        if (((ulong)(((46603 * m1lv + 66882 * m2lv) + 107048004364969L)) >> 16) == upper_bits) {
            long seed = -39761 * m1lv + 35098 * m2lv;
            if ((((ulong)seed) >> 16) == lower_bits) {
                if (((254681119335897L * seed + 120305458776662L) & 0xffffffffffffUL) == original){
        			max_count_1++;
        		}
            }
        }
        if (((ulong)((46603 * (m1lv+1) + 66882 * m2lv) + 107048004364969L) >> 16) == upper_bits) {
        	long seed = -39761 * (m1lv+1)+ 35098 * m2lv;
            if ((((ulong)seed) >> 16) == lower_bits) {
                if (((254681119335897L * seed + 120305458776662L) & 0xffffffffffffUL) == original){
        			max_count_2++;
        		}
            }
        }
        if (((ulong)((46603 * m1lv + 66882 * (m2lv+1)) + 107048004364969L) >> 16) == upper_bits) {
        	long seed = -39761 * m1lv + 35098 * (m2lv +1);
            if ((((ulong)seed) >> 16) == lower_bits) {
                if (((254681119335897L * seed + 120305458776662L) & 0xffffffffffffUL) == original){
        			max_count_3++;
        		}
            }
        }
        if (((ulong)((46603 * (m1lv+1) + 66882 * (m2lv+1)) + 107048004364969L) >> 16) == upper_bits) {
        	long seed = -39761 * (m1lv+1)+ 35098 * (m2lv +1);
            if ((((ulong)seed) >> 16) == lower_bits) {
                if (((254681119335897L * seed + 120305458776662L) & 0xffffffffffffUL) == original){
        			max_count_4++;
        		}
            }
        }
        max_count++;
	}
	debug[id]= max_count_1 <<48 | max_count_2<<32 | max_count_3<<16 | max_count_4;
	ret[id] = max_count;
}
