"""
自对弈 Agent 接口

BaseAgent
  抽象基类，定义 react(seat, events, pending) -> dict

RinshanAgent
  基于 RinshanModel 的主力 AI，通过 GameEncoder 构建 tensor 后调用模型推理

RandomAgent
  随机打牌 agent，用于测试和基线对比
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import torch

from rinshan.tile import Tile
from rinshan.engine.action import (
    ActionType, Action, decode_action,
    chi_type_to_idx, idx_to_chi_type,
)
from rinshan.constants import (
    DISCARD_OFFSET, CHI_OFFSET, PON_OFFSET, DAIMINKAN_OFFSET,
    ANKAN_OFFSET, KAKAN_OFFSET,
    RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN,
    PASS_TOKEN, NUM_TILE_TYPES,
)


# ─────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """
    自对弈 Agent 基类。

    react() 接收当前局面信息，返回 mjai 格式的动作 dict。
    """

    def __init__(self, name: str = "agent"):
        self.name = name

    @abstractmethod
    def react(
        self,
        seat: int,
        player_events: list[dict],
        pending: dict,
    ) -> dict:
        """
        决策接口。

        Args:
            seat:          当前玩家座位（0-3）
            player_events: 该玩家视角的完整事件流（mjai 格式 list[dict]）
            pending:       本次决策描述（含 can_ron/can_chi/can_pon 等字段）

        Returns:
            mjai 格式动作 dict，例如：
              {"type": "dahai",  "actor": 0, "pai": "3m", "tsumogiri": False}
              {"type": "reach",  "actor": 0, "pai": "3m"}
              {"type": "pon",    "actor": 1, "discarder": 0, "pai": "3m",
               "consumed": ["3m", "3m"]}
              {"type": "pass",   "actor": 1}
              {"type": "hora",   "actor": 1, "target": 0, "pai": "3m"}
              {"type": "tsumo",  "actor": 0}
        """

    def on_game_start(self) -> None:
        """可选：游戏开始时的回调"""

    def on_kyoku_end(self) -> None:
        """可选：一局结束时的回调"""

    def on_game_end(self, result: "GameRecord") -> None:  # noqa: F821
        """可选：整场游戏结束时的回调"""


# ─────────────────────────────────────────────
# 随机 Agent（基线 / 测试用）
# ─────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """
    随机打牌 Agent

    打牌：随机选一张手牌打出（不立直，不鸣牌，遇自摸 / 荣和机会时以 50% 概率和牌）
    """

    def __init__(self, name: str = "random", seed: Optional[int] = None):
        super().__init__(name)
        self._rng = random.Random(seed)

    def react(self, seat: int, player_events: list[dict], pending: dict) -> dict:
        ptype = pending.get("type")

        if ptype == "turn_action":
            return self._turn_action(seat, player_events, pending)
        elif ptype == "naki_or_pass":
            return self._naki_or_pass(seat, player_events, pending)
        else:
            return {"type": "pass", "actor": seat}

    def _turn_action(self, seat: int, events: list[dict],
                     pending: dict) -> dict:
        """轮到自己行动（摸牌后）"""
        hand = _extract_hand(events, seat)

        # 自摸和（向听数 -1 时）
        if _tsumo_agari_from_events(events, seat):
            return {"type": "tsumo", "actor": seat}

        # 立直中：只能摸切（打出刚摸的牌）
        if pending.get("in_riichi"):
            last_draw = _last_draw(events, seat)
            if last_draw is not None:
                return {
                    "type": "dahai",
                    "actor": seat,
                    "pai": last_draw.to_mjai(),
                    "tsumogiri": True,
                }
            # fallback（不应出现）
            tile = hand[-1] if hand else Tile.from_mjai("1z")
            return {"type": "dahai", "actor": seat,
                    "pai": tile.to_mjai(), "tsumogiri": True}

        # 随机打一张手牌
        if not hand:
            return {"type": "dahai", "actor": seat, "pai": "1z",
                    "tsumogiri": True}
        tile = self._rng.choice(hand)
        last_draw = _last_draw(events, seat)
        tsumogiri = (last_draw is not None and
                     last_draw.tile_id == tile.tile_id and
                     last_draw.is_aka == tile.is_aka)
        return {
            "type": "dahai",
            "actor": seat,
            "pai": tile.to_mjai(),
            "tsumogiri": tsumogiri,
        }

    def _naki_or_pass(self, seat: int, events: list[dict],
                      pending: dict) -> dict:
        """他家打牌后，决定是否鸣牌"""
        # 50% 概率荣和
        if pending.get("can_ron") and self._rng.random() < 0.5:
            return {
                "type": "hora",
                "actor": seat,
                "target": pending["discarder"],
                "pai": pending["tile"],
            }
        # 其余一律 pass
        return {"type": "pass", "actor": seat}


# ─────────────────────────────────────────────
# Rinshan AI Agent
# ─────────────────────────────────────────────

class RinshanAgent(BaseAgent):
    """
    基于 RinshanModel 的主力 AI Agent

    内部维护与 MjaiSimulator 相同的 GameState，把当前局面编码为模型输入，
    调用 model.react() 得到动作 token，再解码为 mjai 格式 dict 输出。

    libriichi 接口属性（供 RinshanBatchAgent Rust 适配器识别）：
        engine_type = "rinshan"
        name        = agent 名称
    """

    # libriichi py_agent 识别标志
    engine_type: str = "rinshan"

    def __init__(
        self,
        model,           # RinshanModel
        name: str = "rinshan",
        device: str = "cpu",
        temperature: float = 0.8,
        top_p: float = 0.9,
        greedy: bool = False,
    ):
        super().__init__(name)
        self.model = model
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_p = top_p
        self.greedy = greedy

        # 延迟导入（避免循环依赖）
        from rinshan.data.encoder import GameEncoder
        from rinshan.engine.simulator import MjaiSimulator
        self._encoder  = GameEncoder()
        self._sim      = MjaiSimulator()
        # seat -> {"state": GameState, "n_events": int}
        self._state_cache: dict[int, dict[str, object]] = {}

    def on_game_start(self) -> None:
        # 注意：同一个 RinshanAgent 实例可能被 Arena 复用于多场并行对局，
        # 不能在这里清空全局缓存，否则会误伤其他 game_key 下的状态。
        pass

    def on_kyoku_end(self) -> None:
        # 同上：缓存按 game_key 分桶，交由 key 自然隔离。
        pass

    def on_game_end(self, result=None) -> None:
        # 不做全量 clear，避免并行对局互相污染；缓存通过 game_key 区分。
        pass

    def _get_cached_state(self, seat: int, player_events: list[dict], pending: dict):
        game_key = str(pending.get("_game_key", "default"))
        cache_key = (game_key, seat)
        cached = self._state_cache.get(cache_key)

        if cached is None:
            state = _replay_events_to_state(player_events, seat)
        else:
            state = cached["state"]
            n_events = int(cached["n_events"])
            if len(player_events) < n_events:
                state = _replay_events_to_state(player_events, seat)
            elif len(player_events) > n_events:
                state = _advance_state_with_events(state, player_events[n_events:])

        self._state_cache[cache_key] = {"state": state, "n_events": len(player_events)}
        return state

    def react_batch_requests(self, requests: list[tuple[int, list[dict], dict]]) -> list[dict]:
        from rinshan.data.dataset import collate_fn

        responses: list[dict | None] = [None] * len(requests)
        batch_indices: list[int] = []
        batch_encoded: list[dict] = []
        batch_states: list[object] = []
        batch_candidates: list[list[int]] = []
        batch_pending: list[dict] = []
        batch_seats: list[int] = []
        batch_can_tsumo: list[bool] = []

        for i, (seat, player_events, pending) in enumerate(requests):
            ptype = pending.get("type")
            state = self._get_cached_state(seat, player_events, pending)

            if ptype == "turn_action":
                candidates, can_tsumo, _ = _build_turn_candidates(state, seat)
            elif ptype == "naki_or_pass":
                candidates, can_tsumo, _ = _build_naki_candidates(state, seat, pending)
            else:
                responses[i] = {"type": "pass", "actor": seat}
                continue

            if not candidates:
                responses[i] = {"type": "pass", "actor": seat}
                continue

            ann = _state_to_annotation(state, seat, player_events, candidates)
            batch_indices.append(i)
            batch_encoded.append(self._encoder.encode(ann))
            batch_states.append(state)
            batch_candidates.append(candidates)
            batch_pending.append(pending)
            batch_seats.append(seat)
            batch_can_tsumo.append(can_tsumo)

        if batch_encoded:
            encoded = collate_fn(batch_encoded)
            tokens     = encoded["tokens"].to(self.device)
            cand_mask  = encoded["candidate_mask"].to(self.device)
            pad_mask   = encoded["pad_mask"].to(self.device)
            b_tokens   = encoded["belief_tokens"].to(self.device)
            b_pad_mask = encoded["belief_pad_mask"].to(self.device)

            self.model.eval()
            action_idx, q_values = self.model.react(
                tokens, cand_mask, pad_mask,
                b_tokens, b_pad_mask,
                temperature=self.temperature,
                top_p=self.top_p,
                greedy=self.greedy,
            )

            for local_i, orig_i in enumerate(batch_indices):
                candidates = batch_candidates[local_i]
                chosen_token = candidates[action_idx[local_i].item()]
                responses[orig_i] = _token_to_mjai(
                    chosen_token,
                    batch_seats[local_i],
                    batch_states[local_i],
                    batch_pending[local_i],
                    batch_can_tsumo[local_i],
                    q_values=q_values[local_i],
                    candidates=candidates,
                )

        return [r if r is not None else {"type": "pass", "actor": requests[i][0]} for i, r in enumerate(responses)]

    def react(self, seat: int, player_events: list[dict], pending: dict) -> dict:
        return self.react_batch_requests([(seat, player_events, pending)])[0]


# ─────────────────────────────────────────────
# 内部辅助：事件流 → 局面 / 候选 / Annotation
# ─────────────────────────────────────────────

def _extract_hand(events: list[dict], seat: int) -> list[Tile]:
    """从事件流中还原当前手牌"""
    hand: list[Tile] = []
    for ev in events:
        etype = ev.get("type", "")
        if etype == "start_kyoku":
            tehais = ev.get("tehais", [[], [], [], []])
            hand = [Tile.from_mjai(t) for t in tehais[seat] if t != "?"]
        elif etype == "tsumo" and ev.get("actor") == seat:
            pai = ev.get("pai")
            if pai and pai != "?":
                hand.append(Tile.from_mjai(pai))
        elif etype == "dahai" and ev.get("actor") == seat:
            tile = Tile.from_mjai(ev["pai"])
            _safe_remove(hand, tile)
        elif etype in ("chi", "pon", "daiminkan") and ev.get("actor") == seat:
            for t_str in ev.get("consumed", []):
                _safe_remove(hand, Tile.from_mjai(t_str))
        elif etype in ("ankan", "kakan") and ev.get("actor") == seat:
            for t_str in ev.get("consumed", []):
                _safe_remove(hand, Tile.from_mjai(t_str))
            if etype == "kakan":
                pai = ev.get("pai")
                if pai:
                    _safe_remove(hand, Tile.from_mjai(pai))
    return hand


def _last_draw(events: list[dict], seat: int) -> Optional[Tile]:
    """返回该玩家最后一次摸到的牌"""
    for ev in reversed(events):
        if ev.get("type") == "tsumo" and ev.get("actor") == seat:
            pai = ev.get("pai")
            if pai and pai != "?":
                return Tile.from_mjai(pai)
        # 打牌后清空
        if ev.get("type") == "dahai" and ev.get("actor") == seat:
            return None
    return None


def _tsumo_agari_from_events(events: list[dict], seat: int) -> bool:
    """检查当前手牌是否已和牌（自摸）"""
    from rinshan.tile import hand_to_counts
    from rinshan.algo.shanten import calc_shanten
    hand = _extract_hand(events, seat)
    melds = _extract_melds(events, seat)
    if not hand:
        return False
    counts = hand_to_counts(hand)
    return calc_shanten(counts, len(melds)) == -1


def _safe_remove(lst: list[Tile], tile: Tile) -> None:
    for i, t in enumerate(lst):
        if t.tile_id == tile.tile_id and t.is_aka == tile.is_aka:
            lst.pop(i); return
    for i, t in enumerate(lst):
        if t.tile_id == tile.tile_id:
            lst.pop(i); return


def _extract_melds(events: list[dict], seat: int) -> list:
    melds = []
    for ev in events:
        etype = ev.get("type", "")
        if etype in ("chi", "pon", "daiminkan", "ankan", "kakan") \
                and ev.get("actor") == seat:
            consumed = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
            if etype == "kakan":
                pai = ev.get("pai")
                melds.append(("kakan", consumed + ([Tile.from_mjai(pai)] if pai else [])))
            else:
                pai = ev.get("pai")
                melds.append((etype, ([Tile.from_mjai(pai)] if pai else []) + consumed))
    return melds


def _rebuild_state_from_events(events: list[dict], seat: int):
    """
    从 player_events 快速重建 GameState（用于编码器）。
    复用 MjaiSimulator 的事件解析逻辑。
    """
    from rinshan.engine.simulator import MjaiSimulator
    from rinshan.engine.state import GameState
    # 从事件流中找到 start_kyoku
    for ev in events:
        if ev.get("type") == "start_kyoku":
            break

    # 简化：直接用 MjaiSimulator.parse_game 跑一遍事件（但不用结果 Annotations）
    # 保留状态对象
    sim = MjaiSimulator()
    sim._last_state = None

    class _StateCapture:
        def __init__(self):
            self.state = None
    cap = _StateCapture()

    # 手动回放构建 state（避免重复解析）
    # 这里直接调用 simulator 内部方法重建 state
    state = _replay_events_to_state(events, seat)
    return state


def _replay_events_to_state(events: list[dict], pov_seat: int):
    """轻量版 state 重建（专门为自对弈 agent 推理服务）"""
    from rinshan.engine.state import GameState
    from rinshan.engine.simulator import _remove_tile
    from rinshan.tile import Tile
    from rinshan.constants import (
        PROG_DISCARD_BASE, PROG_DRAW_BASE, PROG_RIICHI_BASE,
        PROG_CHI_BASE, PROG_PON_BASE, PROG_DAIMINKAN_BASE,
        PROG_ANKAN_BASE, PROG_KAKAN_BASE, PROG_NEWDORA_BASE,
    )
    from rinshan.engine.action import chi_type_to_idx

    state = GameState()
    for ev in events:
        etype = ev.get("type", "")
        if etype == "start_kyoku":
            bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
            state = GameState(
                round_wind  = bakaze_map.get(ev.get("bakaze", "E"), 0),
                round_num   = ev.get("kyoku", 1),
                honba       = ev.get("honba", 0),
                kyotaku     = ev.get("kyotaku", 0),
                dealer      = ev.get("oya", 0),
                scores      = list(ev.get("scores", [25000]*4)),
                tiles_left  = 70,
                dora_indicators = [Tile.from_mjai(ev["dora_marker"])] if "dora_marker" in ev else [],
                hands       = [[Tile.from_mjai(t) for t in h if t != "?"]
                               for h in ev.get("tehais", [[], [], [], []])],
                discards    = [[] for _ in range(4)],
                melds       = [[] for _ in range(4)],
                riichi_declared = [False]*4,
                riichi_accepted = [False]*4,
                in_riichi       = [False]*4,
                current_player  = ev.get("oya", 0),
                progression     = [],
            )
        elif etype == "tsumo":
            actor = ev.get("actor", 0)
            pai_str = ev.get("pai", "?")
            state.tiles_left -= 1
            state.progression.append(PROG_DRAW_BASE + actor)
            if pai_str != "?":
                tile = Tile.from_mjai(pai_str)
                state.hands[actor].append(tile)
                state.last_draw = tile
                state.current_player = actor
        elif etype == "dahai":
            actor = ev.get("actor", 0)
            tile = Tile.from_mjai(ev["pai"])
            _remove_tile(state.hands[actor], tile)
            state.discards[actor].append(tile)
            state.last_discard = tile
            state.last_draw = None
            # P1 fix: update permanent furiten after a discard.
            # If the discarded tile was in the player's own waits, set furiten.
            if not state.furiten[actor] and not state.in_riichi[actor]:
                from rinshan.tile import hand_to_counts as _htc
                from rinshan.algo.shanten import calc_shanten as _cs
                test = list(_htc(state.hands[actor]))
                test[tile.tile_id] += 1
                if _cs(test, len(state.melds[actor])) == -1:
                    state.furiten[actor] = True
            prog_tok = PROG_DISCARD_BASE + actor * 37 + (
                tile.tile_id if not tile.is_aka
                else {4: 34, 13: 35, 22: 36}[tile.tile_id]
            )
            state.progression.append(prog_tok)
        elif etype == "reach":
            actor = ev.get("actor", 0)
            state.riichi_declared[actor] = True
            state.riichi_discard_idx[actor] = len(state.discards[actor])
            state.progression.append(PROG_RIICHI_BASE + actor)
        elif etype == "reach_accepted":
            actor = ev.get("actor", 0)
            state.riichi_accepted[actor] = True
            state.in_riichi[actor] = True
            state.kyotaku += 1
        elif etype == "dora":
            tile = Tile.from_mjai(ev["dora_marker"])
            state.dora_indicators.append(tile)
            state.progression.append(PROG_NEWDORA_BASE + tile.tile_id)
        elif etype in ("chi", "pon", "daiminkan"):
            actor = ev.get("actor", 0)
            target = ev.get("target", 0)
            pai = Tile.from_mjai(ev["pai"])
            consumed = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
            for t in consumed:
                _remove_tile(state.hands[actor], t)
            state.melds[actor].append((etype, [pai] + consumed))
            if etype == "chi":
                t1, t2 = sorted(consumed, key=lambda t: t.tile_id)
                suit = t1.tile_id // 9
                low  = min(t1.tile_id, t2.tile_id) % 9 + 1
                t_num = pai.number
                form = 0 if t_num == low+2 else (1 if t_num == low+1 else 2)
                state.progression.append(PROG_CHI_BASE + chi_type_to_idx(suit, low, form))
            else:
                base = PROG_PON_BASE if etype == "pon" else PROG_DAIMINKAN_BASE
                state.progression.append(base + actor * 34 + pai.tile_id)
        elif etype in ("ankan", "kakan"):
            actor = ev.get("actor", 0)
            consumed = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
            if etype == "ankan":
                for t in consumed:
                    _remove_tile(state.hands[actor], t)
                state.melds[actor].append(("ankan", consumed))
                state.progression.append(PROG_ANKAN_BASE + actor*34 + consumed[0].tile_id)
            else:
                tile = Tile.from_mjai(ev["pai"])
                _remove_tile(state.hands[actor], tile)
                # P3 fix: 把 melds 里对应的 pon 升级为 kakan
                for i, (mtype, tiles) in enumerate(state.melds[actor]):
                    if mtype == "pon" and tiles[0].tile_id == tile.tile_id:
                        state.melds[actor][i] = ("kakan", tiles + [tile])
                        break
                state.progression.append(PROG_KAKAN_BASE + actor*34 + tile.tile_id)
    return state


def _advance_state_with_events(state, events: list[dict]):
    """基于缓存状态增量回放新增事件。"""
    from rinshan.engine.simulator import _remove_tile
    from rinshan.tile import Tile
    from rinshan.constants import (
        PROG_DISCARD_BASE, PROG_DRAW_BASE, PROG_RIICHI_BASE,
        PROG_CHI_BASE, PROG_PON_BASE, PROG_DAIMINKAN_BASE,
        PROG_ANKAN_BASE, PROG_KAKAN_BASE, PROG_NEWDORA_BASE,
    )
    from rinshan.engine.action import chi_type_to_idx
    from rinshan.tile import hand_to_counts as _htc
    from rinshan.algo.shanten import calc_shanten as _cs

    for ev in events:
        etype = ev.get("type", "")
        if etype == "start_kyoku":
            return _replay_events_to_state(events, 0)
        elif etype == "tsumo":
            actor = ev.get("actor", 0)
            pai_str = ev.get("pai", "?")
            state.tiles_left -= 1
            state.progression.append(PROG_DRAW_BASE + actor)
            if pai_str != "?":
                tile = Tile.from_mjai(pai_str)
                state.hands[actor].append(tile)
                state.last_draw = tile
                state.current_player = actor
        elif etype == "dahai":
            actor = ev.get("actor", 0)
            tile = Tile.from_mjai(ev["pai"])
            _remove_tile(state.hands[actor], tile)
            state.discards[actor].append(tile)
            state.last_discard = tile
            state.last_draw = None
            if not state.furiten[actor] and not state.in_riichi[actor]:
                test = list(_htc(state.hands[actor]))
                test[tile.tile_id] += 1
                if _cs(test, len(state.melds[actor])) == -1:
                    state.furiten[actor] = True
            prog_tok = PROG_DISCARD_BASE + actor * 37 + (
                tile.tile_id if not tile.is_aka else {4: 34, 13: 35, 22: 36}[tile.tile_id]
            )
            state.progression.append(prog_tok)
        elif etype == "reach":
            actor = ev.get("actor", 0)
            state.riichi_declared[actor] = True
            state.riichi_discard_idx[actor] = len(state.discards[actor])
            state.progression.append(PROG_RIICHI_BASE + actor)
        elif etype == "reach_accepted":
            actor = ev.get("actor", 0)
            state.riichi_accepted[actor] = True
            state.in_riichi[actor] = True
            state.kyotaku += 1
        elif etype == "dora":
            tile = Tile.from_mjai(ev["dora_marker"])
            state.dora_indicators.append(tile)
            state.progression.append(PROG_NEWDORA_BASE + tile.tile_id)
        elif etype in ("chi", "pon", "daiminkan"):
            actor = ev.get("actor", 0)
            pai = Tile.from_mjai(ev["pai"])
            consumed = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
            for t in consumed:
                _remove_tile(state.hands[actor], t)
            state.melds[actor].append((etype, [pai] + consumed))
            if etype == "chi":
                t1, t2 = sorted(consumed, key=lambda t: t.tile_id)
                suit = t1.tile_id // 9
                low = min(t1.tile_id, t2.tile_id) % 9 + 1
                t_num = pai.number
                form = 0 if t_num == low + 2 else (1 if t_num == low + 1 else 2)
                state.progression.append(PROG_CHI_BASE + chi_type_to_idx(suit, low, form))
            else:
                base = PROG_PON_BASE if etype == "pon" else PROG_DAIMINKAN_BASE
                state.progression.append(base + actor * 34 + pai.tile_id)
        elif etype in ("ankan", "kakan"):
            actor = ev.get("actor", 0)
            consumed = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
            if etype == "ankan":
                for t in consumed:
                    _remove_tile(state.hands[actor], t)
                state.melds[actor].append(("ankan", consumed))
                state.progression.append(PROG_ANKAN_BASE + actor * 34 + consumed[0].tile_id)
            else:
                tile = Tile.from_mjai(ev["pai"])
                _remove_tile(state.hands[actor], tile)
                for i, (mtype, tiles) in enumerate(state.melds[actor]):
                    if mtype == "pon" and tiles[0].tile_id == tile.tile_id:
                        state.melds[actor][i] = ("kakan", tiles + [tile])
                        break
                state.progression.append(PROG_KAKAN_BASE + actor * 34 + tile.tile_id)
    return state


def _calc_deal_in_risk_oracle(state, seat: int) -> list[float]:
    """
    自对弈 Oracle 版 deal_in_risk 计算。

    自对弈时四家手牌全部可见，直接精确枚举每家对手的待张。
    与 MjaiSimulator._calc_deal_in_risk 逻辑相同，
    独立为函数避免每次实例化 MjaiSimulator。

    返回：risk[tile_id] = 能对该牌荣和的对手数 / 3
          值域 = {0.0, 0.33, 0.67, 1.0}
    """
    from rinshan.tile import hand_to_counts
    from rinshan.algo.shanten import calc_shanten

    danger_count = [0] * 34
    N_OPP = 3

    for opp in range(4):
        if opp == seat:
            continue
        opp_hand = state.hands[opp]
        if not opp_hand:
            continue
        if state.furiten[opp]:
            continue

        meld_count = len(state.melds[opp])
        counts = hand_to_counts(opp_hand)
        n_tiles = sum(counts)

        if n_tiles == 13:
            # 已确定在等待荣和
            if calc_shanten(counts, meld_count) != 0:
                continue
            for tile_id in range(34):
                counts[tile_id] += 1
                if calc_shanten(counts, meld_count) == -1:
                    danger_count[tile_id] += 1
                counts[tile_id] -= 1
        else:
            # 14张（刚摸牌）或副露后手牌数
            if calc_shanten(counts, meld_count) != 0:
                continue
            # 取所有合法打法后的待张并集
            waits: set[int] = set()
            for t_id in range(34):
                if counts[t_id] == 0:
                    continue
                counts[t_id] -= 1
                if calc_shanten(counts, meld_count) == 0:
                    for w in range(34):
                        counts[w] += 1
                        if calc_shanten(counts, meld_count) == -1:
                            waits.add(w)
                        counts[w] -= 1
                counts[t_id] += 1
            for w in waits:
                danger_count[w] += 1

    return [min(1.0, danger_count[t] / N_OPP) for t in range(34)]


def _build_turn_candidates(state, seat: int) -> tuple[list[int], bool, bool]:
    """
    构建打牌决策候选 token 列表。
    返回 (candidates, can_tsumo, can_riichi)
    """
    from rinshan.engine.simulator import MjaiSimulator
    sim = MjaiSimulator()
    cans = sim._compute_discard_candidates(state, seat)
    tokens = sim._build_candidate_tokens(cans, None)
    can_tsumo = cans.can_tsumo
    can_riichi = cans.can_riichi
    return tokens, can_tsumo, can_riichi


def _build_naki_candidates(state, seat: int,
                            pending: dict) -> tuple[list[int], bool, bool]:
    """构建鸣牌候选 token 列表"""
    from rinshan.engine.simulator import MjaiSimulator
    from rinshan.constants import RON_AGARI_TOKEN, PASS_TOKEN
    sim = MjaiSimulator()

    discarder = pending.get("discarder", 0)
    pai = Tile.from_mjai(pending["tile"])
    tokens = sim._compute_naki_candidates(state, seat, discarder, pai)

    # P2 fix: game_board is the authority on whether RON is legal.
    # It tracks junme_furiten and riichi_furiten that GameState does not hold.
    # Reconcile: if game_board says can_ron=False, forcibly remove RON from tokens;
    # if game_board says can_ron=True and compute missed it, add it.
    can_ron = bool(pending.get("can_ron", False))
    if can_ron and RON_AGARI_TOKEN not in tokens:
        tokens = [RON_AGARI_TOKEN] + tokens
    elif not can_ron and RON_AGARI_TOKEN in tokens:
        tokens = [t for t in tokens if t != RON_AGARI_TOKEN]

    if not tokens:
        tokens = [PASS_TOKEN]
    return tokens, False, False


def _state_to_annotation(state, seat: int, events: list[dict],
                          candidates: list[int]):
    """把 GameState + 候选列表打包成 Annotation（用于编码器）"""
    from rinshan.data.annotation import Annotation, AuxTargets
    from rinshan.tile import hand_to_counts
    from rinshan.algo.shanten import calc_shanten

    view = state.player_view(seat)
    counts = hand_to_counts(view.hand)
    sht = calc_shanten(counts, len(view.melds[0]) if view.melds else 0)

    # deal_in_risk: 自对弈中四家手牌全部可见，Oracle 精确计算
    # 天凤日志解析路径由 MjaiSimulator._calc_deal_in_risk 处理（对手不可见时返回全零）
    deal_in_risk = _calc_deal_in_risk_oracle(state, seat)

    aux = AuxTargets(
        shanten     = sht,
        tenpai_prob = float(sht <= 0),
        deal_in_risk= deal_in_risk,
        opp_tenpai  = [int(state.in_riichi[(seat+i+1)%4]) for i in range(3)],
    )

    return Annotation(
        game_id           = "self_play",
        player_id         = seat,
        round_wind        = view.round_wind,
        round_num         = view.round_num,
        honba             = view.honba,
        kyotaku           = view.kyotaku,
        scores            = list(view.scores),
        tiles_left        = view.tiles_left,
        hand              = list(view.hand),
        dora_indicators   = list(view.dora_indicators),
        discards          = [list(d) for d in view.discards],
        melds             = [list(m) for m in view.melds],
        riichi_declared   = list(view.riichi_declared),
        progression       = list(view.progression),
        action_candidates = list(candidates),
        action_chosen     = 0,
        aux               = aux,
    )


def _token_to_mjai(token: int, seat: int, state, pending: dict,
                   can_tsumo: bool,
                   q_values: "Optional[torch.Tensor]" = None,
                   candidates: "Optional[list[int]]" = None) -> dict:
    """把候选 token 解码为 mjai 格式 dict"""
    from rinshan.constants import (
        DISCARD_OFFSET, CHI_OFFSET, PON_OFFSET, DAIMINKAN_OFFSET,
        ANKAN_OFFSET, KAKAN_OFFSET,
        RIICHI_TOKEN, TSUMO_AGARI_TOKEN, RON_AGARI_TOKEN, PASS_TOKEN,
        NUM_TILE_TYPES,
    )

    # 荣和
    if token == RON_AGARI_TOKEN:
        return {
            "type": "hora", "actor": seat,
            "target": pending.get("discarder", seat),
            "pai": pending.get("tile", "1z"),
        }
    # 自摸和
    if token == TSUMO_AGARI_TOKEN:
        return {"type": "tsumo", "actor": seat}
    # Pass / 立直中摸切
    if token == PASS_TOKEN:
        # 立直中 PASS = 摸切（打出刚摸到的牌）
        if pending.get("in_riichi"):
            last = state.last_draw
            if last is not None:
                return {"type": "dahai", "actor": seat,
                        "pai": last.to_mjai(), "tsumogiri": True}
            # fallback：打出手牌最后一张
            hand = state.hands[seat]
            tile = hand[-1] if hand else Tile(0)
            return {"type": "dahai", "actor": seat,
                    "pai": tile.to_mjai(), "tsumogiri": True}
        return {"type": "pass", "actor": seat}
    # 立直（后续跟随打牌）
    if token == RIICHI_TOKEN:
        # 枚举立直后所有合法弃牌，用模型 Q 值选最优
        tile, tsumogiri = _pick_riichi_discard(state, seat, q_values, candidates)
        return {
            "type": "reach", "actor": seat,
            "pai": tile.to_mjai(), "tsumogiri": tsumogiri,
        }
    # 打牌
    if DISCARD_OFFSET <= token < DISCARD_OFFSET + 37:
        idx = token - DISCARD_OFFSET
        if idx < 34:
            tile = Tile(idx)
        else:
            tile = {34: Tile(4, True), 35: Tile(13, True), 36: Tile(22, True)}[idx]
        last = state.last_draw
        tsumogiri = (last is not None and
                     last.tile_id == tile.tile_id and
                     last.is_aka == tile.is_aka)
        return {
            "type": "dahai", "actor": seat,
            "pai": tile.to_mjai(), "tsumogiri": tsumogiri,
        }
    # 吃
    if CHI_OFFSET <= token < CHI_OFFSET + 90:
        chi_idx = token - CHI_OFFSET
        suit, low_num, form = idx_to_chi_type(chi_idx)
        pai_str = pending.get("tile", "1m")
        pai = Tile.from_mjai(pai_str)
        # 推算消耗牌
        hand = state.hands[seat]
        consumed = _find_chi_consumed(hand, suit, low_num, form, pai)
        return {
            "type": "chi", "actor": seat,
            "discarder": pending.get("discarder", 0),
            "pai": pai_str,
            "consumed": [t.to_mjai() for t in consumed],
        }
    # 碰
    if PON_OFFSET <= token < PON_OFFSET + NUM_TILE_TYPES:
        tile_id = token - PON_OFFSET
        pai_str = pending.get("tile", Tile(tile_id).to_mjai())
        hand = state.hands[seat]
        consumed = _find_pon_consumed(hand, tile_id)
        return {
            "type": "pon", "actor": seat,
            "discarder": pending.get("discarder", 0),
            "pai": pai_str,
            "consumed": [t.to_mjai() for t in consumed],
        }
    # 大明杠
    if DAIMINKAN_OFFSET <= token < DAIMINKAN_OFFSET + NUM_TILE_TYPES:
        tile_id = token - DAIMINKAN_OFFSET
        pai_str = pending.get("tile", Tile(tile_id).to_mjai())
        hand = state.hands[seat]
        consumed = _find_daiminkan_consumed(hand, tile_id)
        return {
            "type": "daiminkan", "actor": seat,
            "discarder": pending.get("discarder", 0),
            "pai": pai_str,
            "consumed": [t.to_mjai() for t in consumed],
        }
    # 暗杠
    if ANKAN_OFFSET <= token < ANKAN_OFFSET + NUM_TILE_TYPES:
        tile_id = token - ANKAN_OFFSET
        hand = state.hands[seat]
        consumed = [t for t in hand if t.tile_id == tile_id][:4]
        return {
            "type": "ankan", "actor": seat,
            "consumed": [t.to_mjai() for t in consumed],
        }
    # 加杠
    if KAKAN_OFFSET <= token < KAKAN_OFFSET + NUM_TILE_TYPES:
        tile_id = token - KAKAN_OFFSET
        tile = next((t for t in state.hands[seat] if t.tile_id == tile_id), Tile(tile_id))
        return {"type": "kakan", "actor": seat, "pai": tile.to_mjai()}

    # fallback
    return {"type": "pass", "actor": seat}



def _pick_riichi_discard(
    state,
    seat: int,
    q_values: "Optional[torch.Tensor]",
    candidates: "Optional[list[int]]",
) -> "tuple[Tile, bool]":
    """
    立直时选最优打牌：

    1. 枚举手牌中打出后仍然听牌（shanten==0）的所有合法弃牌。
    2. 在 candidates / q_values 中找到对应的打牌 token，取 Q 值最高的那张。
    3. 若 q_values 不可用（RandomAgent 场景）则回退到第一张合法牌。

    返回 (tile, tsumogiri)。
    """
    from rinshan.tile import hand_to_counts
    from rinshan.algo.shanten import calc_shanten
    from rinshan.constants import DISCARD_OFFSET

    hand = state.hands[seat]
    melds = state.melds[seat]
    last_draw = state.last_draw

    # ── 1. 找出打出后仍听牌的合法牌 ──────────────────────────
    legal_tiles: list[Tile] = []
    seen_ids: set[int] = set()
    for tile in hand:
        if tile.tile_id in seen_ids:
            continue
        seen_ids.add(tile.tile_id)
        test = list(hand)
        # 移除一张（优先普通牌，保留赤）
        for i, t in enumerate(test):
            if t.tile_id == tile.tile_id and not t.is_aka:
                test.pop(i); break
        else:
            for i, t in enumerate(test):
                if t.tile_id == tile.tile_id:
                    test.pop(i); break
        counts = hand_to_counts(test)
        sht = calc_shanten(counts, len(melds))
        if sht == 0:   # 打出后仍然听牌
            legal_tiles.append(tile)

    # 没有合法听牌弃牌（不应发生，fallback 到打最后一张）
    if not legal_tiles:
        tile = hand[-1] if hand else Tile(0)
        tsumogiri = (last_draw is not None and
                     last_draw.tile_id == tile.tile_id and
                     last_draw.is_aka == tile.is_aka)
        return tile, tsumogiri

    # ── 2. 用 Q 值选最优 ──────────────────────────────────────
    best_tile = legal_tiles[0]   # 默认取第一张（fallback）

    if q_values is not None and candidates is not None:
        # 建立 token → Q 值的映射
        token_to_q: dict[int, float] = {}
        for i, cand_token in enumerate(candidates):
            if i < q_values.shape[0]:
                q_val = q_values[i].item()
                if q_val != float('-inf'):
                    token_to_q[cand_token] = q_val

        best_q = float('-inf')
        for tile in legal_tiles:
            # 对应的 discard token（deaka）
            tok = DISCARD_OFFSET + tile.tile_id
            q = token_to_q.get(tok, float('-inf'))
            # 也检查赤宝牌 token（34/35/36）
            if tile.is_aka:
                aka_idx = {4: 34, 13: 35, 22: 36}.get(tile.tile_id)
                if aka_idx is not None:
                    q = max(q, token_to_q.get(DISCARD_OFFSET + aka_idx, float('-inf')))
            if q > best_q:
                best_q = q
                best_tile = tile

    # ── 3. 计算 tsumogiri ────────────────────────────────────
    tsumogiri = (last_draw is not None and
                 last_draw.tile_id == best_tile.tile_id and
                 last_draw.is_aka == best_tile.is_aka)
    return best_tile, tsumogiri


def _find_chi_consumed(hand: list[Tile], suit: int, low_num: int,
                       form: int, pai: Tile) -> list[Tile]:
    """根据吃型推算消耗的手牌（取前两张匹配的）"""
    base = suit * 9
    all3 = [base + low_num - 1, base + low_num, base + low_num + 1]  # 0-indexed
    consumed_ids = [x for x in all3 if x != pai.tile_id]
    result = []
    hand_copy = list(hand)
    for cid in consumed_ids:
        for i, t in enumerate(hand_copy):
            if t.tile_id == cid:
                result.append(t)
                hand_copy.pop(i)
                break
    return result


def _find_pon_consumed(hand: list[Tile], tile_id: int) -> list[Tile]:
    """推算碰消耗的手牌（取前两张匹配的）"""
    result = []
    for t in hand:
        if t.tile_id == tile_id and len(result) < 2:
            result.append(t)
    return result


def _find_daiminkan_consumed(hand: list[Tile], tile_id: int) -> list[Tile]:
    """推算大明杠消耗的手牌（取前三张匹配的）"""
    result = []
    for t in hand:
        if t.tile_id == tile_id and len(result) < 3:
            result.append(t)
    return result
