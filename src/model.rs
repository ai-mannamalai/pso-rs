use log::info;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;

/// A particle is a flat vector of decision variables.
pub type Particle = Vec<f64>;
/// A swarm is a collection of particles.
pub type Population = Vec<Particle>;

/// Eviction policy for the optional objective-function cache.
///
/// `FirstIterator` drops the oldest entry when the cache is full.
/// `Bucket` clears the whole cache when it is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CacheKind {
    #[default]
    FirstIterator,
    Bucket,
}

fn particle_cache_key(p: &[f64]) -> Vec<u64> {
    p.iter().map(|x| x.to_bits()).collect()
}

struct ObjectiveCache {
    map: HashMap<Vec<u64>, f64>,
    order: VecDeque<Vec<u64>>,
    cap: usize,
    kind: CacheKind,
}

impl ObjectiveCache {
    fn new(cap: usize, kind: CacheKind) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap,
            kind,
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }

    fn get(&self, key: &[u64]) -> Option<f64> {
        self.map.get(key).copied()
    }

    fn insert(&mut self, key: Vec<u64>, value: f64) {
        if self.cap == 0 {
            return;
        }
        if let Some(slot) = self.map.get_mut(&key) {
            *slot = value;
            return;
        }
        if self.map.len() >= self.cap {
            match self.kind {
                CacheKind::FirstIterator => {
                    if let Some(old) = self.order.pop_front() {
                        self.map.remove(&old);
                    }
                }
                CacheKind::Bucket => {
                    self.map.clear();
                    self.order.clear();
                }
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
}

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
    cache: Option<Mutex<ObjectiveCache>>,
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
        let cache = match config.cache {
            Some(n) if n > 0 => Some(Mutex::new(ObjectiveCache::new(n, config.cache_kind))),
            _ => None,
        };
        let mut model = Model {
            config,
            flat_dim,
            population,
            population_f_scores,
            x_best,
            f_best: f64::INFINITY,
            seed,
            obj_f,
            cache,
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
        let use_cache = self.cache.is_some();
        if self.config.parallelize {
            self.population_f_scores = self
                .population
                .par_iter()
                .map(|particle| self.eval_particle(particle, flat_dim, dims, use_cache, None))
                .collect();
        } else {
            self.population_f_scores = self
                .population
                .iter()
                .enumerate()
                .map(|(idx, particle)| {
                    self.eval_particle(particle, flat_dim, dims, use_cache, Some(idx))
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

    fn eval_particle(
        &self,
        particle: &Particle,
        flat_dim: usize,
        dims: &[usize],
        use_cache: bool,
        idx: Option<usize>,
    ) -> f64 {
        if self.config.debug {
            if let Some(idx) = idx {
                info!("Evaluating case {} with parameter {:?}", idx, particle);
            }
        }

        if use_cache {
            if let Some(cache) = &self.cache {
                let key = particle_cache_key(particle);
                if let Ok(guard) = cache.lock() {
                    if let Some(hit) = guard.get(&key) {
                        log::debug!("Cache Hit!");
                        return hit;
                    }
                }
                let result = self.obj_f.evaluate(particle, flat_dim, dims);
                if let Ok(mut guard) = cache.lock() {
                    guard.insert(key, result);
                }
                if self.config.debug {
                    if let Some(idx) = idx {
                        info!("Completed case {} with fitness {}", idx, result);
                    }
                }
                return result;
            }
        }

        let result = self.obj_f.evaluate(particle, flat_dim, dims);
        if self.config.debug {
            if let Some(idx) = idx {
                info!("Completed case {} with fitness {}", idx, result);
            }
        }
        result
    }

    /// Best objective value found so far.
    pub fn get_f_best(&self) -> f64 {
        self.f_best
    }

    /// Best position found so far.
    pub fn get_x_best(&self) -> &Particle {
        &self.x_best
    }

    #[cfg(test)]
    fn cache_len(&self) -> Option<usize> {
        self.cache
            .as_ref()
            .and_then(|c| c.lock().ok().map(|g| g.len()))
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
    /// Maximum cached objective evaluations. `None` or `Some(0)` disables the cache.
    pub cache: Option<usize>,
    pub cache_kind: CacheKind,
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
            cache: Some(10_000_000),
            cache_kind: CacheKind::FirstIterator,
            debug: false,
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config {{ dimensions: {:?}, flat_dim: {}, population_size: {}, neighborhood: {}, rho: {}, alpha: {:.4}, lr: {:.4}, c1: {:.4}, c2: {:.4}, bounds: {:?}, t_max: {}, progress_bar: {}, parallelize: {}, record_trajectory: {}, cache: {:?}, debug: {} }}",
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
            self.cache,
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

#[cfg(test)]
mod cache_tests {
    use super::*;
    use rand::SeedableRng;

    fn sphere(p: &Particle, _flat_dim: usize, _dims: &[usize]) -> f64 {
        p.iter().map(|x| x * x).sum()
    }

    #[test]
    fn first_iterator_evicts_oldest() {
        let mut cache = ObjectiveCache::new(2, CacheKind::FirstIterator);
        cache.insert(vec![1], 1.0);
        cache.insert(vec![2], 2.0);
        cache.insert(vec![3], 3.0);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&[1]), None);
        assert_eq!(cache.get(&[2]), Some(2.0));
        assert_eq!(cache.get(&[3]), Some(3.0));
    }

    #[test]
    fn bucket_clears_when_full() {
        let mut cache = ObjectiveCache::new(2, CacheKind::Bucket);
        cache.insert(vec![1], 1.0);
        cache.insert(vec![2], 2.0);
        cache.insert(vec![3], 3.0);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&[1]), None);
        assert_eq!(cache.get(&[2]), None);
        assert_eq!(cache.get(&[3]), Some(3.0));
    }

    #[test]
    fn model_cache_survives_across_evaluations() {
        let config = Config {
            t_max: 1,
            population_size: 1,
            progress_bar: false,
            parallelize: false,
            cache: Some(8),
            cache_kind: CacheKind::FirstIterator,
            ..Config::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut model = Model::new(config, sphere, Some(1), &mut rng);
        model.population[0] = vec![0.5, -0.25];
        model.get_f_values();
        let after_first = model.cache_len();
        model.get_f_values();
        assert_eq!(model.cache_len(), after_first);
        assert!(after_first.unwrap() >= 1);
    }
}
