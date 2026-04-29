use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{BatchAgent, new_py_agent};
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct SelfPlay {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}

#[pymethods]
impl SelfPlay {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }

    /// Run seed_count self-play hanchans with the same Python engine on all 4 seats.
    /// Returns Vec[GameResult]. This is the preferred generation path for RL Stage4,
    /// because one model controls all seats and therefore gets the largest possible
    /// cross-game batch in Rust BatchGame.
    pub fn py_self_play(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<Vec<GameResult>> {
        py.allow_threads(move || {
            self.run_batch(|player_ids| new_py_agent(engine, player_ids), seed_start, seed_count)
        })
    }
}

impl SelfPlay {
    pub fn run_batch<C>(
        &self,
        new_agent: C,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<Vec<GameResult>>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} self-play hanchans",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)
            .map(|seed| (seed, seed_start.1))
            .collect();

        let player_ids: Vec<u8> = [0u8, 1, 2, 3]
            .into_iter()
            .cycle()
            .take(seed_count as usize * 4)
            .collect();

        let mut agents = [new_agent(&player_ids)?];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);

        let mut player_idx = 0usize;
        let indexes: Vec<_> = (0..seed_count as usize)
            .map(|_| {
                std::array::from_fn(|_| {
                    let ret = Index {
                        agent_idx: 0,
                        player_id_idx: player_idx,
                    };
                    player_idx += 1;
                    ret
                })
            })
            .collect();

        let results = batch_game.run(&mut agents, &indexes, &seeds)?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(seed_count)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .try_for_each(|game_result| {
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}.json.gz")]
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log_string()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;

                    anyhow::Ok(())
                })?;
        }

        Ok(results)
    }
}
