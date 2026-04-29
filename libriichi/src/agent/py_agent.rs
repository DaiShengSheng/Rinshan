use super::{BatchAgent, MjaiLogBatchAgent, MortalBatchAgent, RinshanBatchAgent};
use std::str::FromStr;

use anyhow::{Error, Result, bail};
use pyo3::prelude::*;

enum EngineType {
    Mortal,
    MjaiLog,
    Rinshan,
}

impl FromStr for EngineType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mortal" => Ok(Self::Mortal),
            "mjai-log" => Ok(Self::MjaiLog),
            "rinshan" => Ok(Self::Rinshan),
            v => bail!("unknown engine type {v}"),
        }
    }
}

pub fn new_py_agent(engine: PyObject, player_ids: &[u8]) -> Result<Box<dyn BatchAgent>> {
    let engine_type = Python::with_gil(|py| {
        engine
            .bind_borrowed(py)
            .getattr("engine_type")?
            .extract::<&str>()?
            .parse()
    })?;
    let agent = match engine_type {
        EngineType::Mortal => Box::new(MortalBatchAgent::new(engine, player_ids)?) as _,
        EngineType::MjaiLog => Box::new(MjaiLogBatchAgent::new(engine, player_ids)?) as _,
        EngineType::Rinshan => Box::new(RinshanBatchAgent::new(engine, player_ids)?) as _,
    };
    Ok(agent)
}
