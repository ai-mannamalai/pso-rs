use crate::model::*;
use crate::PsoError;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::fmt;
use std::fs::File;
use std::io::Write;

#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};

/// Neighborhood index lists. Gbest does not allocate an n×n table.
#[derive(Clone, Debug)]
pub(crate) enum Neighborhoods {
    Gbest,
    Lbest(Vec<Vec<usize>>),
}

/// Particle Swarm Optimization runner.
pub struct PSO<F: ObjectiveFunction> {
    chi: f64,
    v_max: Vec<f64>,
    pub model: Model<F>,
    neighborhoods: Neighborhoods,
    velocities: Population,
    pub neigh_population: Population,
    pub best_f_values: Vec<f64>,
    pub best_f_trajectory: Vec<f64>,
    pub best_x_trajectory: Vec<Particle>,
    pub seed: Option<u64>,
    rng: ChaCha8Rng,
}

impl<F: ObjectiveFunction> PSO<F> {
    /// Initialize PSO from an already-evaluated model, continuing `rng`.
    pub fn new(model: Model<F>, seed: Option<u64>, mut rng: ChaCha8Rng) -> PSO<F> {
        let phi = model.config.c1 + model.config.c2;
        let chi = constriction_chi(phi);
        let v_max: Vec<f64> = model
            .config
            .bounds
            .iter()
            .map(|(lo, hi)| model.config.alpha * (hi - lo))
            .collect();

        let neighborhoods = create_neighborhoods(
            model.config.population_size,
            model.config.neighborhood_type,
            model.config.rho,
        );

        let last_dim = *model.config.dimensions.last().unwrap_or(&1);
        let mut velocities = Vec::with_capacity(model.config.population_size);
        for _ in 0..model.config.population_size {
            let mut tmp = Vec::with_capacity(model.flat_dim);
            for j in 0..model.flat_dim {
                let vmax = v_max[j % last_dim];
                tmp.push(rng.gen_range(-vmax..vmax));
            }
            velocities.push(tmp);
        }

        let best_f_values = model.population_f_scores.clone();
        let neigh_population = model.population.clone();
        let best_f_trajectory = vec![model.get_f_best()];
        let best_x_trajectory = vec![model.get_x_best().clone()];

        PSO {
            chi,
            v_max,
            model,
            neighborhoods,
            velocities,
            best_f_values,
            neigh_population,
            best_f_trajectory,
            best_x_trajectory,
            seed,
            rng,
        }
    }

    /// Run the swarm until `t_max` function evaluations or `terminate` is true.
    ///
    /// Returns the number of objective evaluations performed during the loop
    /// (initial evaluation is done in [`Model::new`]).
    pub fn run<Term: FnMut(f64) -> bool>(
        &mut self,
        mut terminate: Term,
    ) -> Result<usize, PsoError> {
        #[cfg(feature = "progress")]
        let bar = if self.model.config.progress_bar {
            let bar = ProgressBar::new(self.model.config.t_max as u64);
            if let Ok(style) = ProgressStyle::default_bar()
                .template("{msg} [{elapsed}] {bar:20.cyan/blue} {pos:>7}/{len:7} ETA: {eta}")
            {
                bar.set_style(style);
            }
            Some(bar)
        } else {
            None
        };

        let mut k = 0;
        let pop_size = self.model.config.population_size;
        loop {
            self.update_velocity_and_pos()?;
            self.model.get_f_values();
            self.update_best_positions();

            k += pop_size;
            #[cfg(feature = "progress")]
            if let Some(ref bar) = bar {
                bar.inc(pop_size as u64);
                bar.set_message(format!("{:.6}", self.model.f_best));
            }
            if k >= self.model.config.t_max || terminate(self.model.f_best) {
                break;
            }
        }

        #[cfg(feature = "progress")]
        if let Some(ref bar) = bar {
            bar.finish_and_clear();
        }
        Ok(k)
    }

    fn update_velocity_and_pos(&mut self) -> Result<(), PsoError> {
        let pop_size = self.model.config.population_size;
        let last_dim = *self.model.config.dimensions.last().unwrap_or(&1);
        let lr = self.model.config.lr;
        let c1 = self.model.config.c1;
        let c2 = self.model.config.c2;
        let chi = self.chi;

        for i in 0..pop_size {
            let lbest_i = self.local_best(i);
            for j in 0..self.model.flat_dim {
                let r1: f64 = self.rng.gen_range(0.0..1.0);
                let r2: f64 = self.rng.gen_range(0.0..1.0);

                let x_ij = self.model.population[i][j];
                let pbest_j = self.neigh_population[i][j];
                let lbest_j = self.neigh_population[lbest_i][j];

                let cog = c1 * r1 * (pbest_j - x_ij);
                let soc = c2 * r2 * (lbest_j - x_ij);
                let mut v = chi * (self.velocities[i][j] + cog + soc);

                let bound_index = j % last_dim;
                let vmax = self.v_max[bound_index];
                if v.abs() > vmax {
                    v = v.signum() * vmax;
                }
                self.velocities[i][j] = v;

                let x = x_ij + lr * v;
                if x.is_nan() {
                    return Err(PsoError::NanCoefficient);
                }

                let (lower_bound, upper_bound) = self.model.config.bounds[bound_index];
                if x > upper_bound {
                    self.model.population[i][j] = upper_bound;
                    self.velocities[i][j] = 0.0;
                } else if x < lower_bound {
                    self.model.population[i][j] = lower_bound;
                    self.velocities[i][j] = 0.0;
                } else {
                    self.model.population[i][j] = x;
                }
            }
        }
        Ok(())
    }

    fn update_best_positions(&mut self) {
        for (i, old) in self.best_f_values.iter_mut().enumerate() {
            let new = self.model.population_f_scores[i];
            if new < *old {
                *old = new;
                self.neigh_population[i] = self.model.population[i].clone();
            }
        }
        if self.model.config.record_trajectory {
            self.best_f_trajectory.push(self.model.get_f_best());
            self.best_x_trajectory.push(self.model.get_x_best().clone());
        }
    }

    fn local_best(&self, i: usize) -> usize {
        match &self.neighborhoods {
            Neighborhoods::Gbest => argmin_f(&self.best_f_values).unwrap_or(0),
            Neighborhoods::Lbest(neigh) => {
                let mut best_idx = neigh[i].first().copied().unwrap_or(0);
                let mut best_f = self.best_f_values[best_idx];
                for &idx in &neigh[i][1..] {
                    let f = self.best_f_values[idx];
                    if f < best_f {
                        best_f = f;
                        best_idx = idx;
                    }
                }
                best_idx
            }
        }
    }

    /// Writes the best objective value at each recorded iteration, one per line.
    pub fn write_f_to_file(&self, filepath: &str) -> Result<(), std::io::Error> {
        let mut file = File::create(filepath)?;
        writeln!(
            file,
            "{}",
            self.best_f_trajectory
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )?;
        Ok(())
    }

    /// Writes the best position at each recorded iteration (comma-separated coefficients).
    pub fn write_x_to_file(&self, filepath: &str) -> Result<(), std::io::Error> {
        let mut file = File::create(filepath)?;
        writeln!(
            file,
            "{}",
            self.best_x_trajectory
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|coef| coef.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .collect::<Vec<_>>()
                .join("\n")
        )?;
        Ok(())
    }
}

/// Clerc–Kennedy constriction coefficient. Requires `phi = c1 + c2 >= 4`.
fn constriction_chi(phi: f64) -> f64 {
    let tmp = (phi * phi - 4.0 * phi).sqrt();
    2.0 / (2.0 - phi - tmp).abs()
}

/// Ring of radius `rho` including the particle itself, with modular wrap.
pub(crate) fn create_neighborhoods(
    population_size: usize,
    neighborhood_type: NeighborhoodType,
    rho: usize,
) -> Neighborhoods {
    match neighborhood_type {
        NeighborhoodType::Gbest => Neighborhoods::Gbest,
        NeighborhoodType::Lbest => {
            let n = population_size as i32;
            let rho = rho as i32;
            let mut neighborhoods = Vec::with_capacity(population_size);
            for i in 0..population_size {
                let mut neighbor = Vec::with_capacity((2 * rho + 1) as usize);
                for d in -rho..=rho {
                    let j = ((i as i32 + d) % n + n) % n;
                    neighbor.push(j as usize);
                }
                neighborhoods.push(neighbor);
            }
            Neighborhoods::Lbest(neighborhoods)
        }
    }
}

fn argmin_f(v: &[f64]) -> Option<usize> {
    let mut iter = v.iter().enumerate();
    let (mut best_i, mut best) = iter.next()?;
    for (i, val) in iter {
        if val < best {
            best = val;
            best_i = i;
        }
    }
    Some(best_i)
}

impl<F: ObjectiveFunction> fmt::Display for PSO<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PSO {{ chi: {:.6}, v_max: {:?}, population_size: {}, t_max: {}, lr: {:.4}, c1: {:.4}, c2: {:.4}, f_best: {:.6}, neighborhood: {}, seed: {:?} }}",
            self.chi,
            self.v_max,
            self.model.config.population_size,
            self.model.config.t_max,
            self.model.config.lr,
            self.model.config.c1,
            self.model.config.c2,
            self.model.f_best,
            self.model.config.neighborhood_type,
            self.seed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbest_wraps_at_edges() {
        let neigh = create_neighborhoods(10, NeighborhoodType::Lbest, 2);
        match neigh {
            Neighborhoods::Lbest(n) => {
                assert_eq!(n[0], vec![8, 9, 0, 1, 2]);
                assert_eq!(n[9], vec![7, 8, 9, 0, 1]);
                assert_eq!(n[5], vec![3, 4, 5, 6, 7]);
                assert_eq!(n.len(), 10);
                assert!(n.iter().all(|row| row.iter().all(|&j| j < 10)));
            }
            Neighborhoods::Gbest => panic!("expected lbest"),
        }
    }

    #[test]
    fn gbest_does_not_allocate_index_table() {
        match create_neighborhoods(100, NeighborhoodType::Gbest, 2) {
            Neighborhoods::Gbest => {}
            Neighborhoods::Lbest(_) => panic!("expected gbest"),
        }
    }

    #[test]
    fn argmin_finds_lowest() {
        assert_eq!(argmin_f(&[3.0, 1.0, 2.0]), Some(1));
        assert_eq!(argmin_f(&[]), None);
    }
}
