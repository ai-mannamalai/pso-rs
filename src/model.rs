use log::info;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fmt;

/// A particle is a flat vector of decision variables.
pub type Particle = Vec<f64>;
/// A swarm is a collection of particles.
pub type Population = Vec<Particle>;

/// Objective function evaluated for each particle.
///
/// Closures that match `Fn(&Particle, usize, &[usize]) -> f64` implement this
/// trait automatically. Stateful objectives can implement it on a struct.
pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, p: &Particle, flat_dim: usize, dimensions: &[usize]) -> f64;
}

impl<F> ObjectiveFunction for F
where
    F: Fn(&Particle, usize, &[usize]) -> f64 + Send + Sync,
{
    fn evaluate(&self, p: &Particle, flat_dim: usize, dimensions: &[usize]) -> f64 {
        self(p, flat_dim, dimensions)
    }
}

/// Swarm state: population, scores, and the best position found so far.
pub struct Model<F: ObjectiveFunction> {
    pub config: Config,
    pub flat_dim: usize,
    pub population: Population,
    pub population_f_scores: Vec<f64>,
    pub x_best: Particle,
    pub f_best: f64,
    pub seed: Option<u64>,
    pub obj_f: F,
}

impl<F: ObjectiveFunction> Model<F> {
    /// Creates a new model and evaluates the initial population.
    pub fn new(config: Config, obj_f: F, seed: Option<u64>, rng: &mut ChaCha8Rng) -> Model<F> {
        if config.debug {
            info!("BEGIN: PSO_Model New");
        }

        let dimensions = &config.dimensions;
        let flat_dim: usize = dimensions.iter().product();
        let last_dim = *dimensions.last().unwrap_or(&1);

        let mut population = Vec::with_capacity(config.population_size);
        for _ in 0..config.population_size {
            let mut particle = Vec::with_capacity(flat_dim);
            for flat_i in 0..flat_dim {
                let bound_i = flat_i % last_dim;
                let (lo, hi) = config.bounds[bound_i];
                particle.push(rng.gen_range(lo..hi));
            }
            population.push(particle);
        }

        let population_f_scores = vec![f64::INFINITY; config.population_size];
        let x_best = population[0].clone();
        let mut model = Model {
            config,
            flat_dim,
            population,
            population_f_scores,
            x_best,
            f_best: f64::INFINITY,
            seed,
            obj_f,
        };
        model.get_f_values();
        if model.config.debug {
            info!("END: PSO_Model New");
        }
        model
    }

    /// Evaluates every particle (in parallel when `config.parallelize` is set)
    /// and updates the global best.
    pub fn get_f_values(&mut self) -> &[f64] {
        if self.config.debug {
            info!(
                "BEGIN: get_f_values parallelize:{}, population size:{}",
                self.config.parallelize, self.config.population_size
            );
        }

        let flat_dim = self.flat_dim;
        let dims = self.config.dimensions.as_slice();
        if self.config.parallelize {
            self.population_f_scores = self
                .population
                .par_iter()
                .map(|particle| self.obj_f.evaluate(particle, flat_dim, dims))
                .collect();
        } else {
            self.population_f_scores = self
                .population
                .iter()
                .enumerate()
                .map(|(idx, particle)| {
                    if self.config.debug {
                        info!("Evaluating case {} with parameter {:?}", idx, particle);
                    }
                    let result = self.obj_f.evaluate(particle, flat_dim, dims);
                    if self.config.debug {
                        info!("Completed case {} with fitness {}", idx, result);
                    }
                    result
                })
                .collect();
        }

        let mut f_best = self.f_best;
        let mut x_best_idx: Option<usize> = None;
        for (index, &score) in self.population_f_scores.iter().enumerate() {
            if score < f_best {
                f_best = score;
                x_best_idx = Some(index);
            }
        }
        if let Some(index) = x_best_idx {
            self.x_best = self.population[index].clone();
            self.f_best = f_best;
        }

        if self.config.debug {
            info!("BEST FITNESS: {:?}", self.f_best);
            info!("END: get_f_values");
        }

        &self.population_f_scores
    }

    /// Best objective value found so far.
    pub fn get_f_best(&self) -> f64 {
        self.f_best
    }

    /// Best position found so far.
    pub fn get_x_best(&self) -> &Particle {
        &self.x_best
    }
}

/// Configuration for a PSO run.
#[derive(Clone, Debug)]
pub struct Config {
    pub dimensions: Vec<usize>,
    pub population_size: usize,
    pub neighborhood_type: NeighborhoodType,
    pub rho: usize,
    pub alpha: f64,
    pub c1: f64,
    pub c2: f64,
    pub lr: f64,
    pub bounds: Vec<(f64, f64)>,
    pub t_max: usize,
    pub progress_bar: bool,
    pub parallelize: bool,
    pub record_trajectory: bool,
    pub debug: bool,
}

impl Config {
    pub fn new() -> Config {
        Self::default()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: vec![2],
            population_size: 100,
            neighborhood_type: NeighborhoodType::Lbest,
            rho: 2,
            alpha: 0.1,
            lr: 1.0,
            c1: 2.05,
            c2: 2.05,
            bounds: vec![(-1.0, 1.0); 2],
            t_max: 1000,
            progress_bar: false,
            parallelize: true,
            record_trajectory: false,
            debug: false,
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config {{ dimensions: {:?}, flat_dim: {}, population_size: {}, neighborhood: {}, rho: {}, alpha: {:.4}, lr: {:.4}, c1: {:.4}, c2: {:.4}, bounds: {:?}, t_max: {}, progress_bar: {}, parallelize: {}, record_trajectory: {}, debug: {} }}",
            self.dimensions,
            self.dimensions.iter().product::<usize>(),
            self.population_size,
            self.neighborhood_type,
            self.rho,
            self.alpha,
            self.lr,
            self.c1,
            self.c2,
            self.bounds,
            self.t_max,
            self.progress_bar,
            self.parallelize,
            self.record_trajectory,
            self.debug
        )
    }
}

/// Neighborhood topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborhoodType {
    /// Ring of radius `rho` around each particle.
    Lbest,
    /// Every particle sees the whole swarm.
    Gbest,
}

impl fmt::Display for NeighborhoodType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeighborhoodType::Lbest => write!(f, "Local neighborhood (lbest)"),
            NeighborhoodType::Gbest => write!(f, "Global neighborhood (gbest)"),
        }
    }
}

impl<F: ObjectiveFunction> fmt::Display for Model<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Model {{ dimensions: {:?}, flat_dim: {}, population_size: {}, neighborhood: {}, rho: {}, lr: {:.4}, c1: {:.4}, c2: {:.4}, f_best: {:.6}, seed: {:?}, parallelize: {}, debug: {} }}",
            self.config.dimensions,
            self.flat_dim,
            self.config.population_size,
            self.config.neighborhood_type,
            self.config.rho,
            self.config.lr,
            self.config.c1,
            self.config.c2,
            self.f_best,
            self.seed,
            self.config.parallelize,
            self.config.debug
        )
    }
}
