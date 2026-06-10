use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fmt;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum NumericKind {
    ValueF64(f64),
    ValueBool(bool),
    ValueI32(i32),
}

impl From<f64> for NumericKind {
    fn from(value: f64) -> Self {
        NumericKind::ValueF64(value)
    }
}

impl From<bool> for NumericKind {
    fn from(value: bool) -> Self {
        NumericKind::ValueBool(value)
    }
}

impl From<i32> for NumericKind {
    fn from(value: i32) -> Self {
        NumericKind::ValueI32(value)
    }
}

impl std::fmt::Display for NumericKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strval = match self {
            NumericKind::ValueF64(v) => v.to_string(),
            NumericKind::ValueBool(v) => v.to_string(),
            NumericKind::ValueI32(v) => v.to_string(),
        };
        write!(f, "{}", strval)
    }
}

impl Mul for NumericKind {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        match self {
            NumericKind::ValueF64(v) => NumericKind::ValueF64(v * other.value_f64()),
            NumericKind::ValueBool(v) => NumericKind::ValueBool(v && other.value_bool()),
            NumericKind::ValueI32(v) => NumericKind::ValueI32(v * other.value_i32()),
        }
    }
}

impl Add for NumericKind {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        match self {
            NumericKind::ValueF64(v) => NumericKind::ValueF64(v + other.value_f64()),
            NumericKind::ValueBool(v) => NumericKind::ValueBool(v || other.value_bool()),
            NumericKind::ValueI32(v) => NumericKind::ValueI32(v + other.value_i32()),
        }
    }
}

impl Sub for NumericKind {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        match self {
            NumericKind::ValueF64(v) => NumericKind::ValueF64(v - other.value_f64()),
            NumericKind::ValueBool(v) => NumericKind::ValueBool(v ^ other.value_bool()),
            NumericKind::ValueI32(v) => NumericKind::ValueI32(v - other.value_i32()),
        }
    }
}

pub type Particle = Vec<NumericKind>;
pub type Population = Vec<Particle>;

impl NumericKind {
    pub fn value_f64(&self) -> f64 {
        match self {
            NumericKind::ValueF64(v) => *v,
            NumericKind::ValueBool(v) => *v as i32 as f64,
            NumericKind::ValueI32(v) => *v as f64,
        }
    }

    pub fn value_bool(&self) -> bool {
        match self {
            NumericKind::ValueF64(v) => *v as i32 != 0,
            NumericKind::ValueBool(v) => *v,
            NumericKind::ValueI32(v) => *v != 0,
        }
    }

    pub fn value_i32(&self) -> i32 {
        match self {
            NumericKind::ValueF64(v) => *v as i32,
            NumericKind::ValueBool(v) => *v as i32,
            NumericKind::ValueI32(v) => *v,
        }
    }
}

pub type CastFunctionT = fn(p: &Vec<f64>) -> Vec<NumericKind>;

pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, p: &Particle, flat_dim: usize, dimensions: &[usize]) -> f64;
}

/// Model struct
///
/// It takes in a `Config` instance and `fn` pointer to an objective function and defines a `run` method for running Particle Swarm Optimization.
pub struct Model {
    pub config: Config,
    pub flat_dim: usize,
    pub population: Population,
    pub population_f_scores: Vec<f64>,
    pub x_best: Particle,
    pub f_best: f64,
    pub seed: Option<u64>,
    pub obj_f: Box<dyn ObjectiveFunction>,
    pub cast_f: Option<CastFunctionT>, /* cast Particle -> Particle */
}

impl Model {
    /// Creates a new Model instance
    pub fn new(
        config: Config,
        obj_f: Box<dyn ObjectiveFunction>,
        cast_f: Option<CastFunctionT>,
        seed: Option<u64>,
    ) -> Model {
        let config_debug = config.debug;
        if config_debug {
            println!("BEGIN: PSO_Model New")
        }
        // init population
        let mut rng = thread_rng();
        let mut seeded_rng = ChaCha8Rng::seed_from_u64(0);
        if let Some(seedval) = seed {
            seeded_rng = ChaCha8Rng::seed_from_u64(seedval);
        }
        let mut flat_dim = 1;
        for d in config.dimensions.clone() {
            flat_dim *= d;
        }
        let mut population: Population = vec![];

        for _ in 0..config.population_size {
            let mut particle: Particle = vec![];
            for flat_i in 0..flat_dim {
                let true_i = flat_i % config.dimensions[config.dimensions.len() - 1];
                if seed.is_some() {
                    particle.push(NumericKind::ValueF64(
                        rng.gen_range(config.bounds[true_i].0..config.bounds[true_i].1),
                    ));
                } else {
                    particle.push(NumericKind::ValueF64(
                        seeded_rng.gen_range(config.bounds[true_i].0..config.bounds[true_i].1),
                    ));
                }
            }
            population.push(match cast_f {
                Some(caster) => {
                    let v = particle
                        .iter()
                        .map(|f| (*f).value_f64())
                        .collect::<Vec<f64>>();
                    caster(&v)
                }
                _ => particle,
            });
        }
        let population_f_scores = vec![f64::INFINITY; config.population_size];
        let x_best = population[0].clone();
        let f_best = population_f_scores[0];
        let mut model = Model {
            config,
            flat_dim,
            population,
            population_f_scores,
            x_best,
            f_best,
            seed,
            obj_f,
            cast_f,
        };
        model.get_f_values();
        if config_debug {
            println!("END: PSO_Model New")
        }
        model
    }

    /// Computes the value of the objective function for each particle and updates best found
    ///
    /// Returns the objective function values for all particles
    ///
    /// Uses the rayon crate for parallel computation
    pub fn get_f_values(&mut self) -> Vec<f64> {
        if self.config.debug {
            println!("BEGIN: get_f_values")
        }

        // find the objective function value for each member of the population
        if self.config.parallelize {
            let iter = self.population.par_iter();
            self.population_f_scores = iter
                .map(|particle| {
                    (*self.obj_f).evaluate(particle, self.flat_dim, &self.config.dimensions)
                    // self.population_f_scores[i] = f_score;
                })
                .collect();
        } else {
            let iter = self.population.iter();
            self.population_f_scores = iter
                .map(|particle| {
                    (*self.obj_f).evaluate(particle, self.flat_dim, &self.config.dimensions)
                    // self.population_f_scores[i] = f_score;
                })
                .collect();
        }
        // update best
        let mut f_best = self.f_best;
        let mut x_best = self.x_best.clone();
        for (index, &score) in self.population_f_scores.iter().enumerate() {
            if score < f_best {
                f_best = score;
                x_best = self.population[index].clone();
            }
        }
        self.f_best = f_best;
        self.x_best = x_best;

        if self.config.debug {
            println!("END: get_f_values")
        }

        self.population_f_scores.to_owned()
    }

    /// Returns the best found objective function value
    pub fn get_f_best(&self) -> f64 {
        self.f_best
    }

    /// Returns the best found minimizer
    pub fn get_x_best(&self) -> Particle {
        self.x_best.clone()
    }
}

/// Configuration struct
///
/// Used to define model parameters
#[derive(Debug)]
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
            population_size: 1000,
            neighborhood_type: NeighborhoodType::Lbest,
            rho: 2,
            alpha: 0.1,
            lr: 0.5,
            c1: 2.05,
            c2: 2.05,
            bounds: vec![(-1.0, 1.0); 2],
            t_max: 1000,
            progress_bar: true,
            parallelize: true,
            debug: false,
        }
    }
}

#[derive(Debug)]
pub enum NeighborhoodType {
    Lbest,
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
