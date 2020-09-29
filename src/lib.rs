#[cfg(test)]
mod tests {
    use java_random::{Random, JAVA_LCG};
    use crate::get_next_long;

    #[test]
    fn it_works() {
        let mut r =Random::with_seed(123);
        let value=r.next_long();
        let res=get_next_long((value & 0xffff_ffff_ffff) as u64);
        println!("{:?}", res);
        assert!(res.contains(&(123^JAVA_LCG.multiplier as u64)));
    }
    #[test]
    fn it_works_batch() {
        for i in 0..1000000 {
            let random_value:u64= ((Random::with_seed(i).next_long()) & 0xffff_ffff_ffff) as u64;
            let mut r =Random::with_seed(random_value);
            let value=r.next_long();
            let res=get_next_long((value & 0xffff_ffff_ffff) as u64);
            assert!(res.contains(&((random_value)^JAVA_LCG.multiplier as u64)),"Failed for {}",random_value);
        }

    }
}

fn floor_div(x: i64, y: i64) -> i64 {
    let mut r = x / y;
    // if the signs are different and modulo not zero, round down
    if (x ^ y) < 0 && (r * y != x) {
        r-=1;
    }
    return r;
}

pub fn get_next_long(mut structure_seed: u64) -> Vec<u64> {
    let mut res: Vec<u64> = Vec::with_capacity(2);
    structure_seed = structure_seed & 0xffff_ffff_ffff;
    let lower_bits: i64 = (structure_seed & 0xffff_ffff) as i64;
    let mut upper_bits: i64 = ((structure_seed as u64) >> 32) as i64;
    //Did the lower bits affect the upper bits
    if (lower_bits & 0x8000_0000) != 0 {
        upper_bits += 1; //restoring the initial value of the upper bits
    }

    //The algorithm is meant to have bits_of_danger = 0, but this runs into overflow issues.
    //By using a different small value, we introduce small numerical error which probably cannot break things
    //while keeping everything in range of a long and avoiding nasty BigDecimal/BigInteger overhead
    let bits_of_danger = 1;

    let low_min:i64 = (lower_bits << 16 - bits_of_danger) as i64;
    let low_max:i64 = (((lower_bits + 1) << 16 - bits_of_danger) - 1) as i64;
    let upper_min:i64 = (((upper_bits << 16).wrapping_sub(107048004364969)) >> bits_of_danger) as i64;

    //hardcoded matrix multiplication again
    let m1lv = floor_div(low_max.wrapping_mul(-33441i64).wrapping_add( upper_min.wrapping_mul(17549i64)) , 1i64 << 31 - bits_of_danger) + 1; //I cancelled out a common factor of 2 in this line
    let m2lv = floor_div(low_min.wrapping_mul(46603i64).wrapping_add( upper_min.wrapping_mul( 39761i64)) , 1i64 << 32 - bits_of_danger) + 1;

    //with a lot more effort you can make these loops check 2 things and not 4 but I'm not sure it would even be much faster
    for i in 0..2 {
        for j in 0..2 {
            let seed = (-39761i64.wrapping_mul(m1lv.wrapping_add(i))).wrapping_add( 35098i64.wrapping_mul(m2lv.wrapping_add(j)));
            if ((46603i64.wrapping_mul(m1lv.wrapping_add(i))).wrapping_add( 66882i64.wrapping_mul(m2lv.wrapping_add(j))).wrapping_add(107048004364969i64) >> 16) == upper_bits as i64 {
                if (seed as u64) >> 16 == lower_bits as u64 {
                    res.push((( seed.wrapping_mul(254681119335897).wrapping_add(120305458776662)) & 0xffff_ffff_ffff) as u64)//pull back 2 LCG calls
                }
            }
        }
    }


    return res;
}

