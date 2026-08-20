use pso_rs::*;
use std::process;

const N_PARTICLES: usize = 20;

fn main() {
    let dimensions = vec![N_PARTICLES, 3];
    let config = Config {
        dimensions,
        population_size: 10,
        neighborhood_type: NeighborhoodType::Lbest,
        rho: 2,
        alpha: 0.08,
        lr: 1.0,
        c1: 250.0,
        c2: 0.8,
        bounds: vec![(-2.5, 2.5); 3],
        t_max: N_PARTICLES * 1e5 as usize,
        parallelize: true,
        progress_bar: true,
        record_trajectory: true,
        ..Config::default()
    };
    let before = std::time::Instant::now();
    match pso_rs::run(config, e_lj, |_| false, None) {
        Ok(pso) => {
            println!("Elapsed time: {:.2?}", before.elapsed());
            pso.write_f_to_file("./best_f_trajectory.txt")
                .unwrap_or_else(|err| {
                    eprintln!("Problem writing trajectories: {}.", err);
                    process::exit(1);
                });
            pso.write_x_to_file("./best_x_trajectory.txt")
                .unwrap_or_else(|err| {
                    eprintln!("Problem writing trajectories: {}.", err);
                    process::exit(1);
                });
            let model = pso.model;
            println!("Found minimum: {:#?} ", model.get_f_best());
            println!(
                "Minimizer: {:#?} ",
                reshape(model.get_x_best(), &model.config.dimensions)
            );
        }
        Err(e) => {
            eprintln!("Could not construct PSO: {}", e);
            process::exit(1);
        }
    }
}

fn l2(x_i: &[f64], x_j: &[f64]) -> f64 {
    x_i.iter()
        .zip(x_j)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn v_ij(x_i: &[f64], x_j: &[f64]) -> f64 {
    let denom = 1.0 / l2(x_i, x_j);
    denom.powi(12) - denom.powi(6)
}

fn e_lj(particle: &Particle, _flat_dim: usize, particle_dims: &[usize]) -> f64 {
    let mut sum = 0.0;
    for i in 0..particle_dims[0] - 1 {
        for j in (i + 1)..particle_dims[0] {
            let true_i = i * particle_dims[1];
            let true_j = j * particle_dims[1];
            sum += v_ij(
                &particle[true_i..true_i + particle_dims[1]],
                &particle[true_j..true_j + particle_dims[1]],
            );
        }
    }
    4.0 * sum
}

fn reshape(particle: &Particle, particle_dims: &[usize]) -> Vec<Vec<f64>> {
    let mut reshaped_cluster = vec![];
    let mut i = 0;
    for _ in 0..particle_dims[0] {
        let mut reshaped_molecule = vec![];
        for _ in 0..particle_dims[1] {
            reshaped_molecule.push(particle[i]);
            i += 1;
        }
        reshaped_cluster.push(reshaped_molecule);
    }
    reshaped_cluster
}
