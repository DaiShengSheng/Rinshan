/// RinshanBatchAgent — Rust-side adapter for the Rinshan Python AI engine.
///
/// Unlike MortalBatchAgent (which passes numpy feature matrices), this agent
/// passes raw mjai event JSON + a pending-decision dict directly to the Python
/// method `react_batch_requests(items)`, matching the signature expected by
/// `rinshan.self_play.agent.RinshanAgent`.
///
/// Python engine contract:
/// ```python
/// class RinshanAgent:
///     engine_type: str = "rinshan"
///     name: str
///
///     def react_batch_requests(
///         self,
///         requests: list[tuple[int, list[dict], dict]],
///     ) -> list[dict]:
///         ...
/// ```
///
/// Each request tuple is `(seat, player_events, pending)` where:
/// - `seat`          – player seat 0-3
/// - `player_events` – full mjai event log from this player's PoV (JSON dicts)
/// - `pending`       – decision context dict, e.g.:
///     turn_action:  {"type":"turn_action", "in_riichi": bool, "_game_key": str}
///     naki_or_pass: {"type":"naki_or_pass", "can_ron": bool,
///                    "discarder": int, "tile": str, "_game_key": str}
///
/// The response list contains mjai action dicts:
///   {"type":"dahai", "actor":0, "pai":"3m", "tsumogiri":false}
///   {"type":"reach", "actor":0}
///   {"type":"pon",   "actor":1, "target":0, "pai":"3m", "consumed":["3m","3m"]}
///   {"type":"hora",  "actor":1, "target":0, "pai":"3m"}
///   {"type":"tsumo", "actor":0}
///   {"type":"pass",  "actor":1}
use super::{BatchAgent, InvisibleState};
use crate::arena::GameResult;
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;

use anyhow::{Context, Result};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json as json;

// ─── Tile name translation ────────────────────────────────────────────────────
//
// libriichi uses Tenhou-style honor tile names: E S W N P F C
// Rinshan Python uses mjai-style numeric names:  1z 2z 3z 4z 5z 6z 7z
//
// We translate every tile-string field in the serialised JSON event tree so
// that `Tile.from_mjai()` on the Python side works without modification.

fn libriichi_tile_to_rinshan(s: &str) -> &str {
    match s {
        "E"  => "1z", "S"  => "2z", "W"  => "3z", "N"  => "4z",
        "P"  => "5z", "F"  => "6z", "C"  => "7z",
        // aka (red-five) variants – libriichi uses e.g. "5mr", "5pr", "5sr"
        // which Rinshan also calls "0m", "0p", "0s"
        "5mr" => "0m", "5pr" => "0p", "5sr" => "0s",
        other => other,
    }
}

fn rinshan_tile_to_libriichi(s: &str) -> &str {
    match s {
        "1z" => "E",  "2z" => "S",  "3z" => "W",  "4z" => "N",
        "5z" => "P",  "6z" => "F",  "7z" => "C",
        "0m" => "5mr", "0p" => "5pr", "0s" => "5sr",
        other => other,
    }
}

/// Recursively walk a `serde_json::Value` and translate every string value
/// that looks like a tile name from libriichi format to Rinshan format.
fn translate_tiles_to_rinshan(v: &mut json::Value) {
    match v {
        json::Value::String(s) => {
            let translated = libriichi_tile_to_rinshan(s.as_str());
            if translated != s.as_str() {
                *s = translated.to_owned();
            }
        }
        json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                translate_tiles_to_rinshan(item);
            }
        }
        json::Value::Object(map) => {
            // Only translate fields that are known tile fields.
            // This avoids accidentally mangling "type", "actor", etc.
            const TILE_FIELDS: &[&str] = &[
                "pai", "dora_marker", "bakaze", "consumed",
                "ura_markers", "tile",
            ];
            for (k, val) in map.iter_mut() {
                if TILE_FIELDS.contains(&k.as_str()) {
                    translate_tiles_to_rinshan(val);
                } else if k == "tehais" {
                    translate_tiles_to_rinshan(val);
                }
            }
        }
        _ => {}
    }
}

/// Translate tile names in a Python-response dict from Rinshan → libriichi.
fn translate_tiles_to_libriichi(v: &mut json::Value) {
    match v {
        json::Value::String(s) => {
            let translated = rinshan_tile_to_libriichi(s.as_str());
            if translated != s.as_str() {
                *s = translated.to_owned();
            }
        }
        json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                translate_tiles_to_libriichi(item);
            }
        }
        json::Value::Object(map) => {
            const TILE_FIELDS: &[&str] = &["pai", "consumed"];
            for (k, val) in map.iter_mut() {
                if TILE_FIELDS.contains(&k.as_str()) {
                    translate_tiles_to_libriichi(val);
                }
            }
        }
        _ => {}
    }
}

pub struct RinshanBatchAgent {
    engine: PyObject,
    name: String,
    player_ids: Vec<u8>,

    /// Per-player mjai event log (from that player's PoV).
    /// Index matches `player_ids`.
    logs: Vec<Vec<json::Value>>,

    /// Pending decisions collected during set_scene, consumed in get_reaction.
    /// None means no decision needed for this slot.
    pending: Vec<Option<json::Value>>,

    /// Cached responses from the last batch call.
    /// Indexed by position in player_ids.
    responses: Vec<Option<json::Value>>,

    /// Whether get_reaction has already run the batch for this round.
    evaluated: bool,

    /// Monotonically increasing counter, incremented on each start_game.
    /// Used to make game_key unique across different games played by this agent.
    game_counter: u64,
}

impl RinshanBatchAgent {
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        let name = Python::with_gil(|py| {
            engine
                .bind_borrowed(py)
                .getattr("name")?
                .extract::<String>()
        })?;

        let size = player_ids.len();
        Ok(Self {
            engine,
            name,
            player_ids: player_ids.to_vec(),
            logs: vec![vec![]; size],
            pending: vec![None; size],
            responses: vec![None; size],
            evaluated: false,
            game_counter: 0,
        })
    }

    /// Append one event to all logs, but only the player-visible version:
    /// for Tsumo, hide opponent's pai (replace with "?"); keep everything
    /// else as-is (same simplification used in the Python Arena).
    fn push_event(&mut self, index: usize, log: &[EventExt]) {
        // Rebuild log slice from scratch each scene (simpler and correct).
        // We rebuild from the full log passed by the arena, filtering to the
        // PoV of player_ids[index].
        let seat = self.player_ids[index] as usize;
        self.logs[index] = log
            .iter()
            .filter_map(|ext| {
                let ev = &ext.event;
                // Hide opponent tsumo pai.
                let visible = match ev {
                    Event::Tsumo { actor, pai } if *actor as usize != seat => {
                        let mut v = json::to_value(ev).ok()?;
                        if let Some(o) = v.as_object_mut() {
                            o.insert("pai".to_owned(), json::Value::String("?".to_owned()));
                        }
                        v
                    }
                    // StartKyoku — hide other players' tehais.
                    Event::StartKyoku {
                        bakaze,
                        dora_marker,
                        kyoku,
                        honba,
                        kyotaku,
                        oya,
                        scores,
                        tehais,
                    } => {
                        let masked_tehais: Vec<json::Value> = tehais
                            .iter()
                            .enumerate()
                            .map(|(i, hand)| {
                                if i == seat {
                                    // Translate tile names inside own hand
                                    let tiles: Vec<json::Value> = hand
                                        .iter()
                                        .map(|t| {
                                            json::Value::String(
                                                libriichi_tile_to_rinshan(&t.to_string()).to_owned(),
                                            )
                                        })
                                        .collect();
                                    json::Value::Array(tiles)
                                } else {
                                    // 13 "?" tiles
                                    json::Value::Array(
                                        (0..13)
                                            .map(|_| json::Value::String("?".to_owned()))
                                            .collect(),
                                    )
                                }
                            })
                            .collect();
                        let mut m = json::Map::new();
                        m.insert("type".into(), "start_kyoku".into());
                        m.insert(
                            "bakaze".into(),
                            json::Value::String(
                                libriichi_tile_to_rinshan(&bakaze.to_string()).to_owned(),
                            ),
                        );
                        m.insert(
                            "dora_marker".into(),
                            json::Value::String(
                                libriichi_tile_to_rinshan(&dora_marker.to_string()).to_owned(),
                            ),
                        );
                        m.insert("kyoku".into(), json::Value::Number((*kyoku).into()));
                        m.insert("honba".into(), json::Value::Number((*honba).into()));
                        m.insert("kyotaku".into(), json::Value::Number((*kyotaku).into()));
                        m.insert("oya".into(), json::Value::Number((*oya).into()));
                        m.insert(
                            "scores".into(),
                            json::Value::Array(
                                scores.iter().map(|&s| json::Value::Number(s.into())).collect(),
                            ),
                        );
                        m.insert("tehais".into(), json::Value::Array(masked_tehais));
                        json::Value::Object(m)
                    }
                    _ => {
                        let mut v = json::to_value(ev).ok()?;
                        translate_tiles_to_rinshan(&mut v);
                        v
                    }
                };
                // For the special-cased branches above (Tsumo hidden, StartKyoku)
                // we've already translated inline; for the catch-all branch
                // translation was done inside. But Tsumo (hidden) still needs
                // the non-pai fields translated.
                if matches!(ev, Event::Tsumo { actor, .. } if *actor as usize != seat) {
                    // Only the non-pai fields (actor is a number, no translation needed)
                    // Nothing else to translate in a hidden tsumo.
                }
                Some(visible)
            })
            .collect();
    }

    /// Build the pending dict for turn_action (it's the player's own turn).
    fn build_turn_pending(state: &PlayerState, game_key: &str) -> json::Value {
        let cans = state.last_cans();
        let mut m = json::Map::new();
        m.insert("type".into(), "turn_action".into());
        m.insert(
            "in_riichi".into(),
            json::Value::Bool(state.self_riichi_accepted()),
        );
        m.insert("can_tsumo".into(), json::Value::Bool(cans.can_tsumo_agari));
        m.insert("can_riichi".into(), json::Value::Bool(cans.can_riichi));
        m.insert("can_ankan".into(), json::Value::Bool(cans.can_ankan));
        m.insert("can_kakan".into(), json::Value::Bool(cans.can_kakan));
        m.insert(
            "can_ryukyoku".into(),
            json::Value::Bool(cans.can_ryukyoku),
        );
        m.insert("_game_key".into(), json::Value::String(game_key.to_owned()));
        json::Value::Object(m)
    }

    /// Build the pending dict for naki_or_pass (responding to another player).
    fn build_naki_pending(state: &PlayerState, game_key: &str) -> json::Value {
        let cans = state.last_cans();
        let tile_str = state
            .last_kawa_tile()
            .map(|t| libriichi_tile_to_rinshan(&t.to_string()).to_owned())
            .unwrap_or_else(|| "?".to_owned());
        let discarder = cans.target_actor;

        let mut m = json::Map::new();
        m.insert("type".into(), "naki_or_pass".into());
        m.insert("can_ron".into(), json::Value::Bool(cans.can_ron_agari));
        m.insert("can_pon".into(), json::Value::Bool(cans.can_pon));
        m.insert(
            "can_daiminkan".into(),
            json::Value::Bool(cans.can_daiminkan),
        );
        m.insert(
            "can_chi_low".into(),
            json::Value::Bool(cans.can_chi_low),
        );
        m.insert(
            "can_chi_mid".into(),
            json::Value::Bool(cans.can_chi_mid),
        );
        m.insert(
            "can_chi_high".into(),
            json::Value::Bool(cans.can_chi_high),
        );
        m.insert("discarder".into(), json::Value::Number(discarder.into()));
        m.insert("tile".into(), json::Value::String(tile_str));
        m.insert("_game_key".into(), json::Value::String(game_key.to_owned()));
        json::Value::Object(m)
    }

    /// Run the Python batch inference for all pending slots.
    fn evaluate(&mut self, _game_key: &str) -> Result<()> {
        // Collect all slots that have a pending decision.
        let batch: Vec<(usize, &Vec<json::Value>, &json::Value)> = self
            .pending
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|pv| (i, &self.logs[i], pv)))
            .collect();

        if batch.is_empty() {
            return Ok(());
        }

        let raw_responses: Vec<json::Value> = Python::with_gil(|py| {
            // Build Python list of (seat, events_list, pending_dict) tuples.
            let requests = PyList::empty(py);
            for (i, events, pending) in &batch {
                let seat = self.player_ids[*i] as i64;
                // Convert events to Python list of dicts.
                let py_events = PyList::empty(py);
                for ev in *events {
                    let d = json_value_to_py_dict(py, ev)?;
                    py_events.append(d)?;
                }
                let py_pending = json_value_to_py_dict(py, pending)?;
                let tup = PyTuple::new(py, [
                    seat.into_pyobject(py)?.into_any().unbind(),
                    py_events.into_any().unbind(),
                    py_pending.into_any().unbind(),
                ])?;
                requests.append(tup)?;
            }

            let result = self
                .engine
                .bind_borrowed(py)
                .call_method1(intern!(py, "react_batch_requests"), (requests,))
                .context("react_batch_requests call failed")?;

            // Convert Python list of dicts back to serde_json::Value.
            let py_list = result.downcast::<PyList>().map_err(|e| anyhow::anyhow!("expected list: {:?}", e))?;
            let mut out = Vec::with_capacity(py_list.len());
            for item in py_list.iter() {
                let _s: String = item
                    .call_method0(intern!(py, "__repr__"))
                    .ok()
                    .and_then(|r| r.extract().ok())
                    .unwrap_or_else(|| "{}".to_owned());
                // Use the Python json module to get proper JSON from the dict.
                let json_mod = py.import("json")?;
                let json_str: String = json_mod
                    .call_method1(intern!(py, "dumps"), (item,))?
                    .extract()?;
                let v: json::Value = json::from_str(&json_str)
                    .unwrap_or(json::Value::Object(json::Map::new()));
                out.push(v);
            }
            Ok::<_, anyhow::Error>(out)
        })?;

        // Store responses back by original index.
        for ((i, _, _), resp) in batch.iter().zip(raw_responses.into_iter()) {
            self.responses[*i] = Some(resp);
        }

        Ok(())
    }
}

/// Convert a `serde_json::Value` into a Python `dict` (shallow, recursive).
fn json_value_to_py_dict<'py>(
    py: Python<'py>,
    value: &json::Value,
) -> Result<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    if let json::Value::Object(map) = value {
        for (k, v) in map {
            let py_v = json_value_to_pyobject(py, v)?;
            dict.set_item(k, py_v)?;
        }
    }
    Ok(dict)
}

fn json_value_to_pyobject<'py>(
    py: Python<'py>,
    value: &json::Value,
) -> Result<Bound<'py, PyAny>> {
    Ok(match value {
        json::Value::Null => py.None().into_bound(py),
        json::Value::Bool(b) => {
            let py_bool = b.into_pyobject(py)?;
            <pyo3::Bound<'_, pyo3::types::PyBool> as Clone>::clone(&py_bool).into_any()
        }
        json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py)?.into_any()
            } else {
                n.to_string().into_pyobject(py)?.into_any()
            }
        }
        json::Value::String(s) => s.as_str().into_pyobject(py)?.into_any(),
        json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_pyobject(py, item)?)?;
            }
            list.into_any()
        }
        json::Value::Object(_) => json_value_to_py_dict(py, value)?.into_any(),
    })
}

/// Convert a mjai action dict (from Python) to an `EventExt`.
/// Python returns Rinshan tile names (1z-7z, 0m/0p/0s), translate back to
/// libriichi names (E/S/W/N/P/F/C, 5mr/5pr/5sr) before deserialising.
fn py_dict_to_event(resp: &json::Value, _player_id: u8) -> Result<EventExt> {
    let mut v = resp.clone();
    translate_tiles_to_libriichi(&mut v);
    let event: Event = json::from_value(v).unwrap_or(Event::None);
    Ok(EventExt::no_meta(event))
}

// ─── BatchAgent impl ─────────────────────────────────────────────────────────

impl BatchAgent for RinshanBatchAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn set_scene(
        &mut self,
        index: usize,
        log: &[EventExt],
        state: &PlayerState,
        _invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        self.evaluated = false;
        self.responses[index] = None;

        // Rebuild this player's PoV log.
        self.push_event(index, log);

        if !state.last_cans().can_act() {
            self.pending[index] = None;
            return Ok(());
        }

        // game_key: index encodes which game-slot this player belongs to,
        // making it unique across concurrent games in the same batch.
        let game_key = format!("rust:i{}:{}:{}", index, state.kyoku(), state.honba());

        let cans = state.last_cans();
        let pending = if cans.can_discard
            || cans.can_riichi
            || cans.can_tsumo_agari
            || cans.can_ankan
            || cans.can_kakan
            || cans.can_ryukyoku
        {
            // Player's own turn.
            Self::build_turn_pending(state, &game_key)
        } else {
            // Responding to another player's discard.
            Self::build_naki_pending(state, &game_key)
        };

        self.pending[index] = Some(pending);
        Ok(())
    }

    fn get_reaction(
        &mut self,
        index: usize,
        _log: &[EventExt],
        state: &PlayerState,
        _invisible_state: Option<InvisibleState>,
    ) -> Result<EventExt> {
        if !self.evaluated {
            // game_key is not critical here since it's only used for Python-side
            // state caching; use the same format as set_scene.
            let game_key = format!("rust:{}:{}", state.kyoku(), state.honba());
            self.evaluate(&game_key)?;
            self.evaluated = true;
        }

        // Clear pending for this slot.
        self.pending[index] = None;

        if let Some(resp) = self.responses[index].take() {
            return py_dict_to_event(&resp, self.player_ids[index]);
        }

        // Fallback: pass (should not normally happen).
        Ok(EventExt::no_meta(Event::None))
    }

    fn start_game(&mut self, index: usize) -> Result<()> {
        // Increment counter only when index==0 to count per-game not per-seat.
        if index == 0 {
            self.game_counter += 1;
        }
        // Reset per-game state.
        self.logs[index].clear();
        self.pending[index] = None;
        self.responses[index] = None;
        Ok(())
    }

    fn end_kyoku(&mut self, _index: usize) -> Result<()> {
        Ok(())
    }

    fn end_game(&mut self, _index: usize, _result: &GameResult) -> Result<()> {
        Ok(())
    }
}
