use pso_rs::*;

const N_DIMENSIONS: usize = 3;

fn main() {
    let config = Config {
        dimensions: vec![N_DIMENSIONS],
        population_size: 100,
        bounds: vec![(-10.0, 10.0); N_DIMENSIONS],
        t_max: 1e7 as usize,
        parallelize: true,
        progress_bar: true,
        debug: true,
        ..Config::default()
    };
    let before = std::time::Instant::now();
    let pso = pso_rs::run(config, sum_of_squares, |f_best| f_best < 1e-4, Some(123456)).unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());
    println!("Found minimum: {:#?} ", pso.model.get_f_best());
    println!("Found minimizer: {:#?} ", pso.model.get_x_best());
}

fn sum_of_squares(particle: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
    (0..dimensions[0])
        .map(|i| i as f64 * particle[i].powi(2))
        .sum()
}
