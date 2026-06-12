use core::f64;

use pso_rs::*;

struct Rosenbrock;
impl ObjectiveFunction for Rosenbrock {
    fn evaluate(&self, particle: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| {
                100.0 * ((particle[i + 1] - particle[i]).value_f64().powf(2.0)).powf(2.0)
                    + (1.0 - particle[i].value_f64()).powf(2.0)
            })
            .sum()
    }
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
    let rosenbrock = Box::new(Rosenbrock {});
    let pso = pso_rs::run(config, rosenbrock, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0.into();
    model.population[0][1] = (-2.0).into();
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = NumericKind::ValueF64(1.0);
    model.population[0][1] = NumericKind::ValueF64(1.0);
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
    let pso = pso_rs::run(config, Box::new(Rosenbrock {}), None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = (2.0).into();
    model.population[0][1] = (-2.0).into();
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = (1.0).into();
    model.population[0][1] = (1.0).into();
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
    let pso = pso_rs::run(config, Box::new(Rosenbrock {}), None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = (2.0).into();
    model.population[0][1] = (-2.0).into();
    model.population[0][2] = (-2.0).into();
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = (1.0).into();
    model.population[0][1] = (1.0).into();
    model.population[0][2] = (1.0).into();
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

/// Get potential energy of a cluster of particles
struct Elj;
impl Elj {
    /// Get Euclidian distance of two particles
    fn l2(x_i: Particle, x_j: Particle, particle_dim: usize) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..particle_dim {
            sum += (x_i[i] - x_j[i]).value_f64().powf(2.0);
        }
        sum.sqrt()
    }

    /// Get potential energy of two particles
    fn v_ij(x_i: Particle, x_j: Particle, particle_dim: usize) -> f64 {
        let denom: f64 = 1.0 / Elj::l2(x_i, x_j, particle_dim);
        denom.powf(12.0) - denom.powf(6.0)
    }
}
impl ObjectiveFunction for Elj {
    fn evaluate(&self, particle: &Particle, _flat_dim: usize, particle_dims: &[usize]) -> f64 {
        let mut sum = 0.0;
        for i in 0..particle_dims[0] - 1 {
            for j in (i + 1)..particle_dims[0] {
                let true_i = i * particle_dims[1];
                let true_j = j * particle_dims[1];
                sum += Elj::v_ij(
                    particle[true_i..true_i + particle_dims[1]].to_vec(),
                    particle[true_j..true_j + particle_dims[1]].to_vec(),
                    particle_dims[1],
                );
            }
        }
        4.0 * sum
    }
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
    let e_lj = Box::new(Elj);
    let pso = pso_rs::run(config, e_lj, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = (-0.361635309f64).into();
    model.population[0][1] = 0.0439914505f64.into();
    model.population[0][2] = 0.5828840628f64.into();
    model.population[0][3] = 0.2505889242f64.into();
    model.population[0][4] = 0.6193583398f64.into();
    model.population[0][5] = (-0.161460701f64).into();
    model.population[0][6] = (-0.4082757926f64).into();
    model.population[0][7] = (-0.2212115329f64).into();
    model.population[0][8] = (-0.5067996704f64).into();
    model.population[0][9] = 0.5193221773f64.into();
    model.population[0][10] = (-0.4421382574f64).into();
    model.population[0][11] = 0.0853763087f64.into();
    model.get_f_values();
    assert!(model.get_f_best() < -5.9999999);
}

// solve: arg-min f(x,y,z) = x and { (y-3)*(z-4) } or 1e3
struct BSat;
impl ObjectiveFunction for BSat {
    fn evaluate(&self, p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        assert_eq!(dimensions[0], 3);
        if !p[0].value_bool() {
            1000.0f64
        } else {
            ((p[1].value_f64() - 3.0) * (p[2].value_f64() - 4.0)).abs()
        }
    }
}

fn for_cast_fn(vraw: &Vec<f64>) -> Vec<NumericKind> {
    return vec![
        NumericKind::ValueBool(vraw[0] > 0.0),
        NumericKind::ValueF64(vraw[1]),
        NumericKind::ValueF64(vraw[2]),
    ];
}

#[test]
fn it_computes_boolean_sat_and_roots() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = Config {
        dimensions: vec![3],
        t_max: 100000,
        bounds: vec![(0.0, 1.0), (-5.0, 5.0), (-5.0, 5.0)],
        population_size: 100,
        progress_bar: false,
        parallelize: true,
        debug: true,
        ..Config::default()
    };
    let bsat = Box::new(BSat);
    let pso = pso_rs::run(
        config,
        bsat,
        Some(for_cast_fn),
        Some(|f| f.abs() < 0.0001),
        Some(12345u64),
    )
    .unwrap();
    let model = pso.model;
    let result = (
        model.get_x_best()[0].value_bool(),
        model.get_x_best()[1].value_f64(),
        model.get_x_best()[2].value_f64(),
    );

    log::info!("Best Result: {:?}", result);
    assert!(model.f_best.abs() < 0.01);
    assert_eq!(result.0, true);
    assert!((result.1 - 3.0).abs() < 1.0);
    assert!((result.2 - 4.0).abs() < 1.0);
}
