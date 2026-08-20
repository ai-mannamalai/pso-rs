//! An easy-to-use Particle Swarm Optimization (PSO) implementation.
//!
//! Uses Clerc–Kennedy constriction, optional lbest/gbest neighborhoods, and
//! parallel objective evaluation via [`rayon`].
//!
//! # Examples
//!
//! ## Run PSO
//!
//! ```rust
//! use pso_rs::*;
//!
//! fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
//!     (0..dimensions[0] - 1)
//!         .map(|i| {
//!             let x = p[i];
//!             let y = p[i + 1];
//!             100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2)
//!         })
//!         .sum()
//! }
//!
//! let config = Config {
//!     dimensions: vec![2],
//!     bounds: vec![(-5.0, 10.0); 2],
//!     t_max: 10_000,
//!     progress_bar: false,
//!     ..Config::default()
//! };
//!
//! let pso = pso_rs::run(config, rosenbrock, |f_best| f_best < 1e-4, None).unwrap();
//! println!("Found minimum: {:#?}", pso.model.get_f_best());
//! ```
//!
//! ## Initialize PSO for later execution
//!
//! ```rust
//! use pso_rs::*;
//!
//! fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &[usize]) -> f64 {
//!     (0..dimensions[0] - 1)
//!         .map(|i| {
//!             let x = p[i];
//!             let y = p[i + 1];
//!             100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2)
//!         })
//!         .sum()
//! }
//!
//! let config = Config {
//!     dimensions: vec![2],
//!     bounds: vec![(-5.0, 10.0); 2],
//!     t_max: 10_000,
//!     progress_bar: false,
//!     ..Config::default()
//! };
//!
//! let mut pso = pso_rs::init(config, rosenbrock, None).unwrap();
//! pso.run(|_| false).unwrap();
//! println!("Found minimum: {:#?}", pso.model.get_f_best());
//! println!("Minimizer: {:#?}", pso.model.get_x_best());
//! ```
//!
//! # Notes
//!
//! ## Performance
//!
//! Particles are stored as a flat `Vec<f64>`. Objective values are computed in
//! parallel when [`Config::parallelize`] is `true` (the default).
//!
//! ## Optimization problem dimensionality
//!
//! Multi-dimensional shapes (for example a cluster of 20 molecules in 3D) are
//! specified as `dimensions: vec![20, 3]` and flattened to length 60. Bounds
//! apply to the last axis. You can reshape the best particle after a run:
//!
//! ```rust
//! use pso_rs::*;
//!
//! fn reshape(particle: &Particle, particle_dims: &[usize]) -> Vec<Vec<f64>> {
//!     let mut reshaped_cluster = vec![];
//!     let mut i = 0;
//!     for _ in 0..particle_dims[0] {
//!         let mut reshaped_molecule = vec![];
//!         for _ in 0..particle_dims[1] {
//!             reshaped_molecule.push(particle[i]);
//!             i += 1;
//!         }
//!         reshaped_cluster.push(reshaped_molecule);
//!     }
//!     reshaped_cluster
//! }
//!
//! let config = Config {
//!     dimensions: vec![20, 3],
//!     bounds: vec![(-2.5, 2.5); 3],
//!     t_max: 1,
//!     progress_bar: false,
//!     ..Config::default()
//! };
//!
//! fn dummy(_p: &Particle, _flat_dim: usize, _dimensions: &[usize]) -> f64 {
//!     0.0
//! }
//! let pso = pso_rs::run(config, dummy, |_| false, None).unwrap();
//! println!(
//!     "Best found minimizer: {:#?}",
//!     reshape(pso.model.get_x_best(), &pso.model.config.dimensions)
//! );
//! ```

pub mod model;
pub mod pso;

pub use model::{Config, Model, NeighborhoodType, ObjectiveFunction, Particle, Population};
pub use pso::PSO;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fmt;

/// Error from configuration checks or a NaN update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsoError {
    InvalidConfig(&'static str),
    NanCoefficient,
}

impl fmt::Display for PsoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PsoError::InvalidConfig(msg) => write!(f, "{}", msg),
            PsoError::NanCoefficient => write!(f, "a particle coefficient became NaN"),
        }
    }
}

impl std::error::Error for PsoError {}

/// Creates a model and runs PSO.
///
/// `terminate` is called after each generation with the best fitness so far.
/// Pass `|_| false` to run until `t_max` evaluations. `t_max` counts objective
/// calls performed in the update loop (the initial population is evaluated
/// separately during [`init`]).
pub fn run<F, Term>(
    config: Config,
    obj_f: F,
    terminate: Term,
    seed: Option<u64>,
) -> Result<PSO<F>, PsoError>
where
    F: ObjectiveFunction,
    Term: FnMut(f64) -> bool,
{
    let config_debug = config.debug;
    if config_debug {
        log::info!("BEGIN: run");
    }

    let mut pso = init(config, obj_f, seed)?;
    if config_debug {
        log::info!("END INIT");
        log::info!("BEGIN RUN");
    }
    pso.run(terminate)?;
    if config_debug {
        log::info!("END RUN");
        log::info!("BEST FITNESS: {:?}", pso.model.get_f_best());
        log::info!("BEST SOLUTION: {:?}", pso.model.get_x_best());
    }
    Ok(pso)
}

/// Initializes a PSO instance without running the optimization loop.
pub fn init<F: ObjectiveFunction>(
    config: Config,
    obj_f: F,
    seed: Option<u64>,
) -> Result<PSO<F>, PsoError> {
    let config_debug = config.debug;
    if config_debug {
        log::info!("BEGIN: pso::init");
    }
    assert_config(&config)?;
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let model = Model::new(config, obj_f, seed, &mut rng);
    let pso = PSO::new(model, seed, rng);
    if config_debug {
        log::info!("END: pso::init");
    }
    Ok(pso)
}

fn assert_config(config: &Config) -> Result<(), PsoError> {
    if config.c1 + config.c2 < 4.0 {
        return Err(PsoError::InvalidConfig("c1 + c2 must be at least 4"));
    }
    if config.dimensions.is_empty() {
        return Err(PsoError::InvalidConfig("dimensions must be set"));
    }
    if config.population_size == 0 {
        return Err(PsoError::InvalidConfig(
            "population_size must be at least 1",
        ));
    }
    if config.bounds.is_empty() {
        return Err(PsoError::InvalidConfig("bounds must be set"));
    }
    for &(lo, hi) in &config.bounds {
        if lo >= hi {
            return Err(PsoError::InvalidConfig(
                "each bound must have a strictly lower start than end",
            ));
        }
    }
    let last_dim = config.dimensions[config.dimensions.len() - 1];
    if config.bounds.len() != last_dim {
        return Err(PsoError::InvalidConfig(
            "bounds vector must have the same length as the last dimension of the model",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_zero(_p: &Particle, _flat_dim: usize, _dimensions: &[usize]) -> f64 {
        0.0
    }

    #[test]
    fn rejects_bad_coefficients() {
        let config = Config {
            c1: 1.0,
            c2: 1.0,
            ..Config::default()
        };
        assert!(matches!(
            init(config, constant_zero, None),
            Err(PsoError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_mismatched_bounds() {
        let config = Config {
            dimensions: vec![3],
            bounds: vec![(-1.0, 1.0); 2],
            ..Config::default()
        };
        assert!(matches!(
            init(config, constant_zero, None),
            Err(PsoError::InvalidConfig(_))
        ));
    }
}
