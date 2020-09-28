#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

fn get_next_long(mut structure_seed:u64) ->Vec<u64>{
    let mut res:Vec<u64>=Vec::with_capacity(2);
    structure_seed=structure_seed& 0xffff_ffff_ffff;
    let lower_bits:u32 = structureSeed & 0xffff_ffff;
    let mut upper_bits:u32 = structureSeed >> 32;
    //Did the lower bits affect the upper bits
    if (lower_bits & 0x8000_0000) != 0{
        upper_bits += 1; //restoring the initial value of the upper bits
    }


    //TODO I can only guarantee the algorithm's correctness for bits_of_danger = 0 but believe 1 should still always work, needs to be confirmed!!!

    //The algorithm is meant to have bits_of_danger = 0, but this runs into overflow issues.
    //By using a different small value, we introduce small numerical error which probably cannot break things
    //while keeping everything in range of a long and avoiding nasty BigDecimal/BigInteger overhead
    let bits_of_danger = 1;

    let low_min = lower_bits << 16 - bits_of_danger;
    let low_max = ((lower_bits + 1) << 16 - bits_of_danger) - 1;
    let upper_min = ((upper_bits << 16) - 107048004364969) >> bits_of_danger;

    //hardcoded matrix multiplication again
    let m1lv = Math.floorDiv(low_max * -33441 + upper_min * 17549, 1 << 31 - bits_of_danger) + 1; //I cancelled out a common factor of 2 in this line
    let m2lv = Math.floorDiv(low_min *  46603 + upper_min * 39761, 1 << 32 - bits_of_danger) + 1;

    //with a lot more effort you can make these loops check 2 things and not 4 but I'm not sure it would even be much faster
    for i in 0..2 {
        for j in 0..2 {
            let seed = (-39761 * (m1lv + i) + 35098 * (m2lv + j));
            if ((46603 * (m1lv + i) + 66882 * (m2lv + j)) + 107048004364969 >> 16) == upper_bits {
                if seed >> 16 == lower_bits{
                    res.push((254681119335897 * seed + 120305458776662) & 0xffff_ffff_ffff)//pull back 2 LCG calls
                }
            }
        }
    }


    return res;
}
