use java_random::Random;
use next_long_reverser::get_next_long;

fn main(){
    let mut r =Random::with_seed(123);
    let value=r.next_long();
    let res=get_next_long((value & 0xffff_ffff_ffff) as u64);
    println!("{:?}", res);
}