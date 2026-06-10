use pso_rs::*;

const N_DIMENSIONS: usize = 3;

fn main() {
    let config = Config {
        dimensions: vec![N_DIMENSIONS],
        population_size: 100,
        bounds: vec![(-10.0, 10.0); N_DIMENSIONS],
        t_max: 1e7 as usize,
        parallelize: true,
        progress_bar: false,
        debug: true,
        ..Config::default()
    };
    use std::time::Instant;
    let before = Instant::now();
    let sum_squares = Box::new(SumOfSquares {});
    let pso: pso::PSO = pso_rs::run(
        config,
        sum_squares,
        Some(|f| {
            let mut r: Vec<NumericKind> = vec![];
            for ff in f {
                r.push(NumericKind::ValueF64(*ff));
            }
            r
        }),
        Some(|f_best| f_best < 1e-4),
        Some(123456u64),
    )
    .unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());
    let model = pso.model;
    println!("Found minimum: {:#?} ", model.get_f_best());
    println!("Found minimizer: {:#?} ", model.get_x_best());
}

struct SumOfSquares;
impl ObjectiveFunction for SumOfSquares {
    fn evaluate(&self, particle: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        (0..dimensions[0])
            .map(|i| i as f64 * particle[i].value_f64().powf(2.0))
            .sum()
    }
}
