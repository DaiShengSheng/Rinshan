use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{AkochanAgent, BatchAgent, new_py_agent};
use std::fs::{self, File};
use std::io;
use std::iter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct TwoVsTwo {
    #[pyo3(get)]
    pub disable_progress_bar: bool,
    #[pyo3(get)]
    pub log_dir: Option<String>,
    /// Number of parallel rayon groups. Each group runs its own BatchAgent
    /// instance and its own BatchGame loop, fully isolated from other groups.
    /// Defaults to 1 (original single-threaded behaviour).
    #[pyo3(get)]
    pub parallel_groups: usize,
}

#[pymethods]
impl TwoVsTwo {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None, parallel_groups=1))]
    fn new(
        disable_progress_bar: bool,
        log_dir: Option<String>,
        parallel_groups: usize,
    ) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
            parallel_groups: parallel_groups.max(1),
        }
    }

    pub fn py_vs_py(
        &self,
        challenger: PyObject,
        champion: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<Vec<GameResult>> {
        // `allow_threads` is required, otherwise it will block python GC to
        // run, leading to memory leaks, since this function is doing long
        // tasks.
        let challenger = challenger.clone_ref(py);
        let champion   = champion.clone_ref(py);
        py.allow_threads(move || {
            self.run_batch(
                |player_ids| new_py_agent(challenger.clone_ref(unsafe { Python::assume_gil_acquired() }), player_ids),
                |player_ids| new_py_agent(champion.clone_ref(unsafe { Python::assume_gil_acquired() }), player_ids),
                seed_start,
                seed_count,
            )
        })
    }

    pub fn ako_vs_py(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<()> {
        let engine = engine.clone_ref(py);
        py.allow_threads(move || {
            self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                |player_ids| new_py_agent(engine.clone_ref(unsafe { Python::assume_gil_acquired() }), player_ids),
                seed_start,
                seed_count,
            )?;
            Ok(())
        })
    }

    pub fn py_vs_ako(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<()> {
        let engine = engine.clone_ref(py);
        py.allow_threads(move || {
            self.run_batch(
                |player_ids| new_py_agent(engine.clone_ref(unsafe { Python::assume_gil_acquired() }), player_ids),
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                seed_start,
                seed_count,
            )?;
            Ok(())
        })
    }

    pub fn py_vs_ako_one(
        &self,
        engine: PyObject,
        seed: (u64, u64),
        split: usize,
        py: Python<'_>,
    ) -> Result<()> {
        let engine = engine.clone_ref(py);
        py.allow_threads(move || {
            self.run_one(
                |player_ids| new_py_agent(engine.clone_ref(unsafe { Python::assume_gil_acquired() }), player_ids),
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                seed,
                split,
            )?;
            Ok(())
        })
    }
}

impl TwoVsTwo {
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<Vec<GameResult>>
    where
        C: Fn(&[u8]) -> Result<Box<dyn BatchAgent>> + Send + Sync,
        M: Fn(&[u8]) -> Result<Box<dyn BatchAgent>> + Send + Sync,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} hanchans",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
            seed_count * 2,
        );

        let seeds: Arc<Vec<_>> = Arc::new((seed_start.0..seed_start.0 + seed_count)
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), 2))
            .collect());
        // seeds[i] corresponds to one hanchan.  They come in pairs:
        //   seeds[2k]   → split-A (challenger seats 0,2)
        //   seeds[2k+1] → split-B (challenger seats 1,3)

        // ── Work-stealing across parallel_groups workers ────────────────────
        // Each worker picks one hanchan at a time from the shared cursor,
        // runs it to completion (single-game BatchGame), then picks the next.
        // This ensures:
        //   1. Each game is fully isolated (no cross-game tracker drift).
        //   2. GPU sees a stable batch of ~parallel_groups slots per forward
        //      pass (one slot per worker in flight), saturating better than
        //      a single giant batch where most slots are idle each cycle.
        let n_workers = self.parallel_groups.min(seeds.len()).max(1);
        let cursor    = AtomicUsize::new(0);
        let n_total   = seeds.len();

        // Pre-allocate result slots so workers can write in-place.
        let results: Vec<Mutex<Option<GameResult>>> =
            (0..n_total).map(|_| Mutex::new(None)).collect();
        let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);

        // Outer progress bar: one tick per completed hanchan.
        let bar = if self.disable_progress_bar {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(n_total as u64)
        };
        const TEMPLATE: &str =
            "{spinner:.cyan} {msg}\n[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
        let style = ProgressStyle::with_template(TEMPLATE)
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .tick_chars(".oO°Oo*")
            .progress_chars("#-");
        bar.set_style(style);
        bar.enable_steady_tick(Duration::from_millis(150));

        rayon::scope(|s| {
            for _ in 0..n_workers {
                let nc  = &new_challenger_agent;
                let nm  = &new_champion_agent;
                let cur = &cursor;
                let res = &results;
                let err = &first_error;
                let bar = &bar;

                let seeds_ref = Arc::clone(&seeds);
                s.spawn(move |_| {
                    loop {
                        let idx = cur.fetch_add(1, Ordering::Relaxed);
                        if idx >= n_total { break; }

                        let is_split_b = idx % 2 == 1;
                        let ch_ids: &[u8] = if is_split_b { &[1, 3] } else { &[0, 2] };
                        let bl_ids: &[u8] = if is_split_b { &[0, 2] } else { &[1, 3] };

                        let mut agents: [Box<dyn BatchAgent>; 2] = match (|| {
                            Ok::<_, anyhow::Error>([nc(ch_ids)?, nm(bl_ids)?])
                        })() {
                            Ok(a)  => a,
                            Err(e) => { *err.lock().unwrap() = Some(e); break; }
                        };

                        let indexes = if is_split_b {
                            [[
                                Index { agent_idx: 1, player_id_idx: 0 },
                                Index { agent_idx: 0, player_id_idx: 0 },
                                Index { agent_idx: 1, player_id_idx: 1 },
                                Index { agent_idx: 0, player_id_idx: 1 },
                            ]]
                        } else {
                            [[
                                Index { agent_idx: 0, player_id_idx: 0 },
                                Index { agent_idx: 1, player_id_idx: 0 },
                                Index { agent_idx: 0, player_id_idx: 1 },
                                Index { agent_idx: 1, player_id_idx: 1 },
                            ]]
                        };

                        let batch_game = BatchGame::tenhou_hanchan(true);
                        match batch_game.run(&mut agents, &indexes, &[seeds_ref[idx]]) {
                            Ok(mut r) => {
                                *res[idx].lock().unwrap() = r.pop();
                                bar.inc(1);
                                let done = idx + 1;
                                let secs = bar.elapsed().as_secs_f64().max(0.001);
                                bar.set_message(format!(
                                    "{n_workers} workers  {:.2} hanchan/s",
                                    done as f64 / secs,
                                ));
                            }
                            Err(e) => { *err.lock().unwrap() = Some(e); break; }
                        }
                    }
                });
            }
        });
        bar.abandon();

        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        let results: Vec<GameResult> = results
            .into_iter()
            .filter_map(|m| m.into_inner().unwrap())
            .collect();

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(seed_count * 2)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .enumerate()
                .try_for_each(|(i, game_result)| {
                    let split_name = ["a", "b"][i % 2];
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]
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

    pub fn run_one<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed: (u64, u64),
        split: usize, // must be within 0..2
    ) -> Result<GameResult>
    where
        C: Fn(&[u8]) -> Result<Box<dyn BatchAgent>> + Send + Sync,
        M: Fn(&[u8]) -> Result<Box<dyn BatchAgent>> + Send + Sync,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: {} w/ {:#x}, split: {}, start 1 hanchan",
            seed.0,
            seed.1,
            split
        );

        let challenger_player_ids = if split == 0 { [0, 2] } else { [1, 3] };
        let champion_player_ids = if split == 0 { [1, 3] } else { [0, 2] };

        let mut agents = [
            new_challenger_agent(&challenger_player_ids)?,
            new_champion_agent(&champion_player_ids)?,
        ];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);

        let indexes = if split == 0 {
            [[
                Index {
                    agent_idx: 0,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 1,
                },
            ]]
        } else {
            [[
                Index {
                    agent_idx: 1,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 1,
                },
            ]]
        };

        let results = batch_game.run(&mut agents, &indexes, &[seed])?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let split_name = ["a", "b"][split];
            let (seed, key) = seed;
            let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]
                .iter()
                .collect();

            let log = results[0].dump_json_log_string()?;
            let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
            let mut f = File::create(filename)?;
            io::copy(&mut comp, &mut f)?;
        }

        Ok(results.into_iter().next().unwrap())
    }
}
