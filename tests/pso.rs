use pso_rs::*;

fn rosenbrock(particle: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
    (0..dimensions[0] - 1)
        .map(|i| {
            let x = particle[i];
            let y = particle[i + 1];
            100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2)
        })
        .sum()
}

fn sphere(particle: &Particle, _flat_dim: usize, _dimensions: &[usize]) -> f64 {
    particle.iter().map(|x| x * x).sum()
}

#[test]
fn it_runs_non_parallel() {
    let config = Config {
        t_max: 1,
        population_size: 1,
        progress_bar: false,
        parallelize: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, |_| false, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0;
    model.population[0][1] = -2.0;
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = 1.0;
    model.population[0][1] = 1.0;
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_rosenbrock_2d() {
    let config = Config {
        t_max: 1,
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, |_| false, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0;
    model.population[0][1] = -2.0;
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = 1.0;
    model.population[0][1] = 1.0;
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_rosenbrock_3d() {
    let config = Config {
        dimensions: vec![3],
        t_max: 1,
        bounds: vec![(-5.0, 10.0); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, |_| false, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0;
    model.population[0][1] = -2.0;
    model.population[0][2] = -2.0;
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = 1.0;
    model.population[0][1] = 1.0;
    model.population[0][2] = 1.0;
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
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

#[test]
fn it_computes_correct_minimum_e_lj() {
    let config = Config {
        dimensions: vec![4, 3],
        bounds: vec![(-2.5, 2.5); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, e_lj, |_| false, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = -0.361635309;
    model.population[0][1] = 0.0439914505;
    model.population[0][2] = 0.5828840628;
    model.population[0][3] = 0.2505889242;
    model.population[0][4] = 0.6193583398;
    model.population[0][5] = -0.161460701;
    model.population[0][6] = -0.4082757926;
    model.population[0][7] = -0.2212115329;
    model.population[0][8] = -0.5067996704;
    model.population[0][9] = 0.5193221773;
    model.population[0][10] = -0.4421382574;
    model.population[0][11] = 0.0853763087;
    model.get_f_values();
    assert!(model.get_f_best() < -5.9999999);
}

/// arg-min f(x,y,z) = 0 when first coordinate is "true" (x > 0) and (y-3)(z-4) = 0
fn bsat(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
    assert_eq!(dimensions[0], 3);
    if p[0] <= 0.0 {
        1000.0
    } else {
        ((p[1] - 3.0) * (p[2] - 4.0)).abs()
    }
}

#[test]
fn it_computes_boolean_sat_and_roots() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = Config {
        dimensions: vec![3],
        t_max: 100_000,
        bounds: vec![(0.0, 1.0), (-5.0, 5.0), (-5.0, 5.0)],
        population_size: 100,
        progress_bar: false,
        parallelize: true,
        debug: true,
        ..Config::default()
    };
    let pso = pso_rs::run(config, bsat, |f| f.abs() < 0.0001, Some(12345)).unwrap();
    let model = pso.model;
    let result = (
        model.get_x_best()[0],
        model.get_x_best()[1],
        model.get_x_best()[2],
    );

    log::info!("Best Result: {:?}", result);
    assert!(model.f_best.abs() < 0.01);
    assert!(result.0 > 0.0);
    assert!((result.1 - 3.0).abs() < 1.0 || (result.2 - 4.0).abs() < 1.0);
}

#[test]
fn same_seed_is_deterministic() {
    let make = || {
        let config = Config {
            t_max: 400,
            population_size: 20,
            progress_bar: false,
            parallelize: false,
            ..Config::default()
        };
        pso_rs::run(config, sphere, |_| false, Some(42)).unwrap()
    };
    let a = make();
    let b = make();
    assert_eq!(a.model.get_x_best(), b.model.get_x_best());
    assert_eq!(a.model.get_f_best(), b.model.get_f_best());
}

/// Same seed must replay init *and* the update RNG (positions and velocities).
#[test]
fn seeded_rng_replays_full_swarm() {
    let config = || Config {
        t_max: 200,
        population_size: 16,
        dimensions: vec![3],
        bounds: vec![(-2.0, 2.0); 3],
        progress_bar: false,
        parallelize: false,
        record_trajectory: false,
        ..Config::default()
    };

    let mut a = pso_rs::init(config(), sphere, Some(0xC0FFEE)).unwrap();
    let mut b = pso_rs::init(config(), sphere, Some(0xC0FFEE)).unwrap();
    assert_eq!(a.seed, Some(0xC0FFEE));
    assert_eq!(a.model.population, b.model.population);
    assert_eq!(a.model.population_f_scores, b.model.population_f_scores);

    a.run(|_| false).unwrap();
    b.run(|_| false).unwrap();
    assert_eq!(a.model.population, b.model.population);
    assert_eq!(a.model.population_f_scores, b.model.population_f_scores);
    assert_eq!(a.neigh_population, b.neigh_population);
    assert_eq!(a.best_f_values, b.best_f_values);
    assert_eq!(a.model.get_x_best(), b.model.get_x_best());
}

/// Regression: `None` used to draw from ChaCha8 seeded at 0, so it matched `Some(0)`.
#[test]
fn unseeded_rng_is_not_seed_zero() {
    let config = || Config {
        t_max: 1,
        population_size: 12,
        progress_bar: false,
        parallelize: false,
        ..Config::default()
    };
    let unseeded = pso_rs::init(config(), sphere, None).unwrap();
    let seed_zero = pso_rs::init(config(), sphere, Some(0)).unwrap();
    assert_ne!(unseeded.model.population, seed_zero.model.population);
}

#[test]
fn different_seeds_diverge() {
    let config = || Config {
        t_max: 1,
        population_size: 8,
        progress_bar: false,
        parallelize: false,
        ..Config::default()
    };
    let a = pso_rs::init(config(), sphere, Some(1)).unwrap();
    let b = pso_rs::init(config(), sphere, Some(2)).unwrap();
    assert_ne!(a.model.population, b.model.population);
}

#[test]
fn it_finds_sphere_minimum() {
    let config = Config {
        dimensions: vec![2],
        bounds: vec![(-5.0, 5.0); 2],
        t_max: 20_000,
        population_size: 30,
        progress_bar: false,
        parallelize: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, sphere, |f| f < 1e-4, Some(123)).unwrap();
    assert!(
        pso.model.get_f_best() < 1e-3,
        "expected a near-zero sphere minimum, got {}",
        pso.model.get_f_best()
    );
}
