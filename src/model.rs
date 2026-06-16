use log::{debug, info};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Index, Mul, Sub};
use std::sync::{Arc, RwLock};

pub trait KeyValueTrait {
    // 1. Associated Type
    type Key: std::hash::Hash + Eq + Clone + std::fmt::Display;
    type Value: PartialOrd + Clone + Copy + std::fmt::Display;

    // 2. Identification Method
    fn key(&self) -> Self::Key;
    fn value(&self) -> Self::Value;
}

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

pub type CastFunctionT = fn(p: &Vec<f64>) -> Particle;

pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, p: &Particle, flat_dim: usize, dimensions: &[usize]) -> f64;
}

//TODO: Implement LRU, LFU, etc. Cache types.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum CacheKind {
    FirstIterator,
    Bucket,
}

#[derive(Clone, Debug)]
pub struct ConcurrentGenericCache<T: KeyValueTrait + std::fmt::Display> {
    inner: Arc<RwLock<HashMap<T::Key, T::Value>>>,
    size: usize,
    cache_kind: Option<CacheKind>,
}

// Type alias for convenience: Always uses String as the key
pub type StringKeyValueTraitCache<T> = ConcurrentGenericCache<T>;

impl<T: KeyValueTrait + std::fmt::Display> ConcurrentGenericCache<T> {
    pub fn new(cache_size: usize, cache_kind: Option<CacheKind>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            size: cache_size,
            cache_kind,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        log::debug!("Len[Self] = {}", self.inner.read().unwrap().len());
        self.inner.read().unwrap().len()
    }

    pub fn insert(&mut self, item: T) -> Option<T::Value> {
        let curr_size = self.len();
        if curr_size >= self.size {
            match self.cache_kind {
                Some(CacheKind::FirstIterator) => {
                    let key = {
                        let mut handle = self.inner.write();
                        let binding = handle.as_mut();
                        let writeable = binding.unwrap();
                        #[allow(clippy::map_clone)]
                        let elements = writeable.keys().map(|k| k.clone()).collect::<Vec<_>>();
                        (*elements.index(0)).clone()
                    };

                    let mut handle = self.inner.write();
                    let binding = handle.as_mut();
                    let writeable = binding.unwrap();
                    writeable.remove(&key);
                }
                Some(CacheKind::Bucket) | None => {
                    let mut handle = self.inner.write();
                    let binding = handle.as_mut();
                    let writeable = binding.unwrap();
                    writeable.clear();
                }
            };
        }

        self.inner.write().ok()?.insert(item.key(), item.value());
        Some(item.value())
    }

    pub fn get(&self, key: &T::Key) -> Option<T::Value>
    where
        T: Clone,
    {
        debug!("Get for ");
        self.inner.read().ok()?.get(key).cloned()
    }

    pub fn remove(&self, key: &T::Key) -> Option<T::Value> {
        self.inner.write().ok()?.remove(key)
    }
}

#[derive(Clone)]
pub struct ArgForObjectiveFunction(Particle, f64);
impl ArgForObjectiveFunction {
    fn new(particle: &Particle) -> Self {
        ArgForObjectiveFunction(particle.clone(), f64::MAX)
    }
}
impl fmt::Display for ArgForObjectiveFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format vector elements using their Display impl and include the score
        let elems = self
            .0
            .iter()
            .map(|n| format!("{}", n))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "ArgForObjectiveFunction([{}], {})", elems, self.1)
    }
}
impl KeyValueTrait for ArgForObjectiveFunction {
    type Key = String;
    type Value = f64;
    fn value(&self) -> f64 {
        self.1
    }
    fn key(&self) -> Self::Key {
        // Create a unique key representation of the tuple data
        // Example: converting the vector and float into a single string
        format!("{:?}", self.0)
    }
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
            info!("BEGIN: PSO_Model New")
        }
        // init population
        let mut rng = thread_rng();
        let mut seeded_rng = ChaCha8Rng::seed_from_u64(0);
        if let Some(seedval) = seed {
            seeded_rng = ChaCha8Rng::seed_from_u64(seedval);
        }
        let dimensions = &config.dimensions;
        let flat_dim: usize = dimensions.iter().product();
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
        if seed.is_some() {
            population.swap(seeded_rng.gen_range(0..config.population_size), 0);
        } else {
            population.swap(rng.gen_range(0..config.population_size), 0);
        };
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
            info!("END: PSO_Model New")
        }
        model
    }

    /// Computes the value of the objective function for each particle and updates best found
    ///
    /// Returns the objective function values for all particles
    ///
    /// Uses the rayon crate for parallel computation
    pub fn get_f_values(&mut self) -> Vec<f64> {
        type StringCacheObj = StringKeyValueTraitCache<ArgForObjectiveFunction>;
        let mut cache = StringCacheObj::new(
            self.config.cache.unwrap_or_default(),
            Some(CacheKind::FirstIterator),
        );

        if self.config.debug {
            log::info!(
                "BEGIN: get_f_values parallelize:{}, population size:{}, len : {}",
                self.config.parallelize,
                self.config.population_size,
                self.population.len()
            )
        }

        // find the objective function value for each member of the population
        if self.config.parallelize {
            let iter = self.population.par_iter();
            self.population_f_scores = iter
                .map(|particle| {
                    (*self.obj_f).evaluate(particle, self.flat_dim, &self.config.dimensions)
                })
                .collect();
        } else {
            let iter = self.population.iter();
            self.population_f_scores = iter
                .enumerate()
                .map(|(idx, particle)| {
                    if self.config.debug {
                        log::info!("Evaluating case {} with parameter {:?}", idx, particle);
                    }

                    let arg: &ArgForObjectiveFunction = &ArgForObjectiveFunction::new(particle);
                    if let Some(result) = cache.get(&arg.key().to_string()) {
                        log::debug!("Cache Hit!");
                        result
                    } else {
                        let result = (*self.obj_f).evaluate(
                            particle,
                            self.flat_dim,
                            &self.config.dimensions,
                        );
                        if self.config.debug {
                            log::info!("Completed case {} with fitness {}", idx, result);
                        }
                        let arg: ArgForObjectiveFunction =
                            ArgForObjectiveFunction(particle.clone(), result);
                        cache.insert(arg);
                        result
                    }
                })
                .collect();
        }

        if self.config.debug {
            log::info!("    Completed evaluation; sorting for best scores");
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
            log::info!("    Completed sorting scores");
            log::info!("    BEST X: {:?}", self.x_best);
            log::info!("    BEST FITNESS: {:?}", self.f_best);
            log::info!("END: get_f_values");
        }

        self.population_f_scores.clone()
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
    pub cache: Option<usize>,
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
            cache: Some(1000_0000_usize),
            debug: log::log_enabled!(log::Level::Debug) || log::log_enabled!(log::Level::Info),
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config {{ dimensions: {:?}, flat_dim_heuristic: {}, population_size: {}, neighborhood: {}, rho: {}, alpha: {:.4}, lr: {:.4}, c1: {:.4}, c2: {:.4}, bounds: {:?}, t_max: {}, progress_bar: {}, parallelize: {}, debug: {} }}",
            self.dimensions,
            // flat_dim isn't stored here; show product-of-dimensions heuristic
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
            self.debug
        )
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

impl fmt::Display for Model {
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
#[cfg(test)]
mod test {
    use crate::NumericKind;
    use crate::*;

    #[test]
    fn basic_cache_testing() {
        type StringCacheObj = StringKeyValueTraitCache<ArgForObjectiveFunction>;
        for cache_kind in vec![
            Some(CacheKind::FirstIterator),
            Some(CacheKind::Bucket),
            None,
        ] {
            let mut cache: ConcurrentGenericCache<ArgForObjectiveFunction> =
                StringCacheObj::new(2, cache_kind);
            let raw_key = vec![NumericKind::ValueF64(40.0)];
            let item = ArgForObjectiveFunction::new(&raw_key);
            assert_eq!(cache.inner.read().unwrap().get(&item.key()), None);
            assert_eq!(cache.inner.read().unwrap().is_empty(), true);
            for value in 0..10 {
                let raw_key = vec![NumericKind::ValueF64(40.0 * value as f64)];
                let item = ArgForObjectiveFunction {
                    0: raw_key,
                    1: (314159.0 * (value as f64)),
                };
                println!("Calling insert with {}", item);
                cache.insert(item.clone());
                assert_eq!(cache.get(&item.key()), Some(&item.value()).copied());
                assert!(cache.len() <= 10);
            }
            assert_eq!(cache.len(), 2);
        }
    }
}
