use core::f64;

use pso_rs::*;

#[test]
fn it_runs_non_parallel() {
    fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| {
                100.0 * ((p[i + 1] - p[i]).value_f64().powf(2.0)).powf(2.0)
                    + (1.0 - p[i].value_f64()).powf(2.0)
            })
            .sum()
    }

    let config = Config {
        t_max: 1,
        population_size: 1,
        progress_bar: false,
        parallelize: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = NumericKind::ValueF64(2.0);
    model.population[0][1] = NumericKind::ValueF64(-2.0);
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = NumericKind::ValueF64(1.0);
    model.population[0][1] = NumericKind::ValueF64(1.0);
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_rosenbrock_2d() {
    fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| {
                100.0 * ((p[i + 1] - p[i]).value_f64().powf(2.0)).powf(2.0)
                    + (1.0 - p[i].value_f64()).powf(2.0)
            })
            .sum()
    }

    let config = Config {
        t_max: 1,
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = NumericKind::ValueF64(2.0);
    model.population[0][1] = NumericKind::ValueF64(-2.0);
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = NumericKind::ValueF64(1.0);
    model.population[0][1] = NumericKind::ValueF64(1.0);
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_rosenbrock_3d() {
    fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| {
                100.0 * ((p[i + 1] - p[i]).value_f64().powf(2.0)).powf(2.0)
                    + (1.0 - p[i].value_f64()).powf(2.0)
            })
            .sum()
    }

    let config = Config {
        dimensions: vec![3],
        t_max: 1,
        bounds: vec![(-5.0, 10.0); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = NumericKind::ValueF64(2.0);
    model.population[0][1] = NumericKind::ValueF64(-2.0);
    model.population[0][2] = NumericKind::ValueF64(-2.0);
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = NumericKind::ValueF64(1.0);
    model.population[0][1] = NumericKind::ValueF64(1.0);
    model.population[0][2] = NumericKind::ValueF64(1.0);
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_e_lj() {
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
        let denom: f64 = 1.0 / l2(x_i, x_j, particle_dim);
        denom.powf(12.0) - denom.powf(6.0)
    }

    /// Get potential energy of a cluster of particles
    fn e_lj(particle: &Particle, _flat_dim: usize, particle_dims: &[usize]) -> f64 {
        let mut sum = 0.0;
        for i in 0..particle_dims[0] - 1 {
            for j in (i + 1)..particle_dims[0] {
                let true_i = i * particle_dims[1];
                let true_j = j * particle_dims[1];
                sum += v_ij(
                    particle[true_i..true_i + particle_dims[1]].to_vec(),
                    particle[true_j..true_j + particle_dims[1]].to_vec(),
                    particle_dims[1],
                );
            }
        }
        4.0 * sum
    }
    let config = Config {
        dimensions: vec![4, 3],
        bounds: vec![(-2.5, 2.5); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };

    let pso = pso_rs::run(config, e_lj, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = NumericKind::ValueF64(-0.361635309);
    model.population[0][1] = NumericKind::ValueF64(0.0439914505);
    model.population[0][2] = NumericKind::ValueF64(0.5828840628);
    model.population[0][3] = NumericKind::ValueF64(0.2505889242);
    model.population[0][4] = NumericKind::ValueF64(0.6193583398);
    model.population[0][5] = NumericKind::ValueF64(-0.161460701);
    model.population[0][6] = NumericKind::ValueF64(-0.4082757926);
    model.population[0][7] = NumericKind::ValueF64(-0.2212115329);
    model.population[0][8] = NumericKind::ValueF64(-0.5067996704);
    model.population[0][9] = NumericKind::ValueF64(0.5193221773);
    model.population[0][10] = NumericKind::ValueF64(-0.4421382574);
    model.population[0][11] = NumericKind::ValueF64(0.0853763087);
    model.get_f_values();
    assert!(model.get_f_best() < -5.9999999);
}

#[test]
fn it_computes_boolean_sat_and_roots() {
    fn bsat(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
        assert_eq!(dimensions[0], 3);
        if !p[0].value_bool() {
            f64::MAX
        } else {
            //(x-3)*(y-4)
            ((p[1].value_f64() - 3.0) * (p[2].value_f64() - 4.0)).abs()
        }
    }

    let config = Config {
        dimensions: vec![3],
        t_max: 1,
        bounds: vec![(0.0, 1.0), (-5.0, 5.0), (-5.0, 5.0)],
        population_size: 3,
        progress_bar: true,
        ..Config::default()
    };
    let pso = pso_rs::run(config, bsat, None, None, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = NumericKind::ValueBool(false);
    model.population[0][1] = NumericKind::ValueF64(-2.0);
    model.population[0][2] = NumericKind::ValueF64(-2.0);

    model.population[1][0] = NumericKind::ValueBool(false);
    model.population[1][1] = NumericKind::ValueF64(-2.0);
    model.population[1][2] = NumericKind::ValueF64(2.0);

    model.population[2][0] = NumericKind::ValueBool(true);
    model.population[2][1] = NumericKind::ValueF64(2.0);
    model.population[2][2] = NumericKind::ValueF64(-2.0);
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);
}
