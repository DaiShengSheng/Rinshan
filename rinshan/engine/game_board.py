"""
GameBoard — 纯 Python 麻将牌局引擎（自对弈用）

功能：
  - 管理完整牌局状态（发牌 / 摸牌 / 打牌 / 鸣牌 / 杠 / 立直 / 和了 / 流局）
  - 把游戏事件广播为 mjai 格式 dict 序列
  - 支持种子化随机（可复现）

本模块是纯局面推进，不包含任何 AI 决策逻辑。
规则遵循天凤凤凰桌：
  - 赤宝牌 (5m/5p/5s 各 1 张)
  - 无三家荣和流局
  - 天和/地和不叠加其他役，恒为一倍役满
  - 满贯切上
  - 飞出立即结束
"""
from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from rinshan.tile import Tile


# ─────────────────────────────────────────────
# 常量 & 辅助
# ─────────────────────────────────────────────

_YAOCHUHAI = frozenset([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])

# 每局总张数
_TOTAL_TILES = 136
# 岭上牌 4 张 + 宝牌指示牌 5 张 + 里宝牌 5 张
_RINSHAN_COUNT = 4
_DORA_COUNT    = 5
_URA_COUNT     = 5


def _build_full_deck() -> list[Tile]:
    """生成含赤宝牌的 136 张完整牌组"""
    deck: list[Tile] = []
    # 万子/饼子/索子：各 9 种 × 4 张，其中 5m/5p/5s 有 1 张赤
    for suit_base in (0, 9, 18):
        for num in range(9):
            tile_id = suit_base + num
            is_5 = (num == 4)  # 第 5 张（0-indexed: 4）
            for copy in range(4):
                is_aka = is_5 and (copy == 0)
                deck.append(Tile(tile_id, is_aka))
    # 字牌：7 种 × 4 张，无赤
    for tile_id in range(27, 34):
        for _ in range(4):
            deck.append(Tile(tile_id, False))
    assert len(deck) == 136
    return deck


def _seed_to_rng(nonce: int, key: int, kyoku: int, honba: int) -> random.Random:
    """用 SHA-3 从 nonce/key/kyoku/honba 生成确定性随机数"""
    h = hashlib.sha3_256()
    h.update(struct.pack("<QQ BB", nonce, key, kyoku, honba))
    seed = int.from_bytes(h.digest()[:8], "little")
    return random.Random(seed)


class GameEndReason(IntEnum):
    NONE    = 0
    HORA    = 1      # 和了
    RYUKYOKU= 2      # 流局（荒牌 / 途中流局）
    TOBI    = 3      # 飞出（点数 < 0）


@dataclass
class KyokuResult:
    """一局的结算结果"""
    kyoku:   int
    honba:   int
    can_renchan:          bool
    has_hora:             bool
    has_abortive_ryukyoku:bool
    kyotaku_left:         int
    deltas:               list[int]   # 四家得点变化（不含供托）
    scores_after:         list[int]   # 结算后四家点数（含供托归并）


# ─────────────────────────────────────────────
# 核心：单局状态机
# ─────────────────────────────────────────────

class KyokuBoard:
    """
    驱动一局麻将对局的状态机。

    外部循环：
        board = KyokuBoard(kyoku, honba, kyotaku, scores, seed)
        while True:
            if board.need_action:
                action = agent.react(board.seat_needing_action, board.get_player_events(seat))
                board.push_action(action)
            else:
                done = board.step()
                if done:
                    break
        result = board.result
    """

    def __init__(
        self,
        kyoku: int,
        honba: int,
        kyotaku: int,
        scores: list[int],
        game_seed: tuple[int, int],
    ):
        self.kyoku   = kyoku
        self.honba   = honba
        self.kyotaku = kyotaku
        self.scores  = list(scores)

        # ── 洗牌 ──────────────────────────────
        rng = _seed_to_rng(game_seed[0], game_seed[1], kyoku, honba)
        deck = _build_full_deck()
        rng.shuffle(deck)

        # 按天凤惯例分配
        # 配牌 13×4
        self.haipai: list[list[Tile]] = [
            list(deck[i*13:(i+1)*13]) for i in range(4)
        ]
        idx = 52
        # 岭上牌（4 张，pop 取）
        self._rinshan: list[Tile] = list(deck[idx:idx+4]); idx += 4
        # 宝牌指示牌（5 张，pop 取；初始只翻第一张）
        self._dora_wall: list[Tile] = list(deck[idx:idx+5]); idx += 5
        # 里宝牌（5 张，立直和了时翻）
        self._ura_wall: list[Tile] = list(deck[idx:idx+5]); idx += 5
        # 摸牌山（70 张，pop 取）
        self._yama: list[Tile] = list(deck[idx:idx+70]); idx += 70
        assert idx == 136

        # ── 场面状态 ──────────────────────────
        self.oya  = kyoku % 4          # 庄家绝对座位
        self.dora_indicators: list[Tile] = [self._dora_wall.pop()]
        self.tiles_left = 70

        # 四家局内状态
        self.hands:          list[list[Tile]] = [list(h) for h in self.haipai]
        self.discards:       list[list[Tile]] = [[] for _ in range(4)]
        self.melds:          list[list]       = [[] for _ in range(4)]
        self.riichi:         list[bool]       = [False] * 4
        self.riichi_junme:   list[int]        = [-1] * 4
        self.furiten:        list[bool]       = [False] * 4
        # Permanent riichi furiten: after riichi, player discarded a tile in their waits
        self._riichi_furiten: list[bool]      = [False] * 4
        # Temporary same-junme furiten: an opponent discarded a winning tile this junme
        # but the player chose not to ron.  Cleared when the player draws their next tile.
        self._junme_furiten: list[bool]       = [False] * 4

        # Ippatsu: True between riichi declaration and the player's next draw/discard
        # or any intervening naki.  Reset on draw, naki, or after the player discards.
        self._ippatsu: list[bool]             = [False] * 4

        # Chankan: set to True in _handle_kakan so that when _ask_naki detects
        # a ron from an eligible player, is_chankan=True is passed to hora calc.
        self._chankan_tile: Optional[Tile]    = None  # non-None during kakan reaction window

        self.tsumo_actor = self.oya
        self.last_discard: Optional[Tile] = None   # Bug fix: initialize last_discard
        self._from_rinshan   = False
        self._pending_dora   = False  # 杠后待翻宝牌
        self._pending_dora_tsumo = False  # kakan 后待翻（在下次摸牌前翻）
        self._riichi_to_accept: Optional[int] = None
        self.kans       = 0           # 全场杠数
        self._check_four_kan = False  # 四家各杠 < 4，等待下一个打牌来触发四杠散了

        # 流局相关
        self._can_nagashi     = [True] * 4
        self._can_four_wind   = True
        self._four_wind_tile: Optional[Tile] = None
        self._four_riichi     = 0     # 四家立直
        self._accepted_riichi = 0

        # 包牌
        self._paos: list[Optional[int]] = [None] * 4

        # ── 事件日志（mjai 格式，广播给所有玩家）──
        self._log: list[dict] = []
        # 每个玩家视角的可见事件（手牌 + 公开事件）
        self._player_log: list[list[dict]] = [[] for _ in range(4)]

        # ── 决策请求 ──────────────────────────
        # 当游戏需要某玩家做出动作时，设置此字段
        self.pending_decisions: list[Optional[dict]] = [None] * 4
        # 玩家的响应（None 表示尚未响应）
        self._reactions: list[Optional[dict]] = [None] * 4

        # ── 结果 ─────────────────────────────
        self.result: Optional[KyokuResult] = None
        self._done   = False

        # 初始化：发送 start_kyoku 事件，然后进入庄家摸牌等待
        self._emit_start_kyoku()
        # 给庄家多摸 1 张（配牌后庄家 13 张，立刻摸 1 张成 14 张）
        first_tile = self._draw_from_yama()
        self.hands[self.oya].append(first_tile)  # 把牌实际加入手牌
        self.last_draw = first_tile
        self.tsumo_actor = self.oya
        self._emit_tsumo(self.oya, first_tile)
        self._ask_discard_or_win(self.oya, is_tsumo=True)

    # ──────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done

    def get_log(self) -> list[dict]:
        """完整 mjai 事件流（旁观者视角）"""
        return list(self._log)

    def get_player_log(self, seat: int) -> list[dict]:
        """指定座位的可见事件流（手牌隐藏其他家）"""
        return list(self._player_log[seat])

    def has_pending(self) -> bool:
        """是否有玩家需要决策"""
        return any(d is not None for d in self.pending_decisions)

    def push_reaction(self, seat: int, action: dict) -> None:
        """
        提交 seat 玩家的决策（mjai 格式 dict）

        当所有需要决策的玩家都提交后，调用 resolve() 推进游戏。
        """
        assert self.pending_decisions[seat] is not None, \
            f"Seat {seat} has no pending decision"
        self._reactions[seat] = action

    def ready_to_resolve(self) -> bool:
        """所有有 pending_decisions 的玩家都已提交响应"""
        for seat in range(4):
            if self.pending_decisions[seat] is not None \
               and self._reactions[seat] is None:
                return False
        return True

    def resolve(self) -> None:
        """
        处理所有玩家的响应，推进游戏状态，直到下一个决策点或游戏结束。
        调用前必须 ready_to_resolve() == True。
        """
        assert self.ready_to_resolve(), "Not all reactions submitted"
        self._process_reactions()

    # ──────────────────────────────────────────
    # 事件发射辅助
    # ──────────────────────────────────────────

    def _broadcast(self, event: dict) -> None:
        """广播公开事件到所有视角"""
        self._log.append(event)
        for seat in range(4):
            self._player_log[seat].append(event)

    def _broadcast_private(self, event: dict, seat: int) -> None:
        """私有事件（如摸牌）只加入当事玩家视角，公开日志用隐藏版本"""
        self._log.append(event)
        self._player_log[seat].append(event)
        # 其他玩家看到的是 pai="?" 的版本
        hidden = {**event, "pai": "?"}
        for s in range(4):
            if s != seat:
                self._player_log[s].append(hidden)

    def _emit_start_kyoku(self) -> None:
        bakaze_str = ["E", "S", "W", "N"][self.kyoku // 4]
        kyoku_num  = self.oya + 1   # 1-based，配牌局=庄家序号
        event = {
            "type":        "start_kyoku",
            "bakaze":      bakaze_str,
            "kyoku":       kyoku_num,
            "honba":       self.honba,
            "kyotaku":     self.kyotaku,
            "oya":         self.oya,
            "dora_marker": self.dora_indicators[0].to_mjai(),
            "scores":      list(self.scores),
            "tehais":      [
                [t.to_mjai() for t in self.hands[s]] for s in range(4)
            ],
        }
        # 发到各玩家视角时，隐藏对手手牌
        self._log.append(event)
        for seat in range(4):
            visible = {**event, "tehais": [
                [t.to_mjai() for t in self.hands[s]] if s == seat else ["?"] * 13
                for s in range(4)
            ]}
            self._player_log[seat].append(visible)

    def _emit_tsumo(self, seat: int, tile: Tile) -> None:
        event = {"type": "tsumo", "actor": seat, "pai": tile.to_mjai()}
        self._broadcast_private(event, seat)

    def _emit_dahai(self, seat: int, tile: Tile, tsumogiri: bool) -> None:
        event = {"type": "dahai", "actor": seat, "pai": tile.to_mjai(),
                 "tsumogiri": tsumogiri}
        self._broadcast(event)

    def _emit_reach(self, seat: int) -> None:
        self._broadcast({"type": "reach", "actor": seat})

    def _emit_reach_accepted(self, seat: int) -> None:
        self._broadcast({"type": "reach_accepted", "actor": seat})

    def _emit_chi(self, actor: int, target: int, pai: Tile,
                  consumed: list[Tile]) -> None:
        self._broadcast({
            "type": "chi", "actor": actor, "target": target,
            "pai": pai.to_mjai(),
            "consumed": [t.to_mjai() for t in consumed],
        })

    def _emit_pon(self, actor: int, target: int, pai: Tile,
                  consumed: list[Tile]) -> None:
        self._broadcast({
            "type": "pon", "actor": actor, "target": target,
            "pai": pai.to_mjai(),
            "consumed": [t.to_mjai() for t in consumed],
        })

    def _emit_daiminkan(self, actor: int, target: int, pai: Tile,
                        consumed: list[Tile]) -> None:
        self._broadcast({
            "type": "daiminkan", "actor": actor, "target": target,
            "pai": pai.to_mjai(),
            "consumed": [t.to_mjai() for t in consumed],
        })

    def _emit_ankan(self, actor: int, consumed: list[Tile]) -> None:
        self._broadcast({
            "type": "ankan", "actor": actor,
            "consumed": [t.to_mjai() for t in consumed],
        })

    def _emit_kakan(self, actor: int, pai: Tile) -> None:
        self._broadcast({
            "type": "kakan", "actor": actor, "pai": pai.to_mjai(),
        })

    def _emit_dora(self, tile: Tile) -> None:
        self._broadcast({"type": "dora", "dora_marker": tile.to_mjai()})

    def _emit_hora(self, actor: int, target: int,
                   deltas: list[int], ura_markers: list[Tile]) -> None:
        self._broadcast({
            "type": "hora", "actor": actor, "target": target,
            "deltas": list(deltas),
            "ura_markers": [t.to_mjai() for t in ura_markers],
        })

    def _emit_ryukyoku(self, reason: str, deltas: list[int],
                       tenpai: list[bool]) -> None:
        self._broadcast({
            "type": "ryukyoku",
            "reason": reason,
            "deltas": list(deltas),
            "tenpai": list(tenpai),
        })

    def _emit_end_kyoku(self) -> None:
        self._broadcast({"type": "end_kyoku"})

    # ──────────────────────────────────────────
    # 决策请求
    # ──────────────────────────────────────────

    def _clear_pending(self) -> None:
        self.pending_decisions = [None] * 4
        self._reactions        = [None] * 4

    def _ask_discard_or_win(self, seat: int, is_tsumo: bool) -> None:
        """向 seat 玩家询问：打牌 / 立直 / 自摸和 / 九种流局 / 暗杠 / 加杠"""
        self._clear_pending()
        self.pending_decisions[seat] = {
            "type": "turn_action",
            "seat": seat,
            "is_tsumo": is_tsumo,
            "in_riichi": self.riichi[seat],   # agent 需要知道是否立直，以限制打牌选项
        }

    def _ask_naki(self, discarder: int, tile: Tile,
                    is_chankan: bool = False) -> None:
        """Ask all eligible players whether to naki/ron the discarded tile.

        is_chankan=True when called from _handle_kakan (chankan opportunity).
        """
        self._clear_pending()
        self._chankan_tile = tile if is_chankan else None

        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten

        for seat in range(4):
            if seat == discarder:
                continue
            if self.riichi[seat]:
                # In-riichi: can only ron (not chi/pon/daiminkan)
                if self._can_ron(seat, discarder, tile):
                    self.pending_decisions[seat] = {
                        "type": "naki_or_pass",
                        "seat": seat,
                        "discarder": discarder,
                        "tile": tile.to_mjai(),
                        "can_ron": True,
                        "can_chi": False,
                        "can_pon": False,
                        "can_daiminkan": False,
                        "is_chankan": is_chankan,
                    }
                else:
                    # Riichi player is eligible (is_wait) but cannot ron (furiten).
                    # Set junme (same-round) furiten: they missed their winning tile.
                    test_counts = hand_to_counts(self.hands[seat])
                    test_counts[tile.tile_id] += 1
                    is_wait = calc_shanten(test_counts, len(self.melds[seat])) == -1
                    if is_wait:
                        self._junme_furiten[seat] = True
            else:
                can_ron      = self._can_ron(seat, discarder, tile)
                can_chi      = (self._can_chi(seat, discarder, tile)
                                if not is_chankan else False)  # no chi on chankan
                can_pon      = (self._can_pon(seat, tile)
                                if not is_chankan else False)  # no pon on chankan
                can_daiminkan= (self._can_daiminkan(seat, tile)
                                if not is_chankan else False)
                if can_ron or can_chi or can_pon or can_daiminkan:
                    self.pending_decisions[seat] = {
                        "type": "naki_or_pass",
                        "seat": seat,
                        "discarder": discarder,
                        "tile": tile.to_mjai(),
                        "can_ron": can_ron,
                        "can_chi": can_chi,
                        "can_pon": can_pon,
                        "can_daiminkan": can_daiminkan,
                        "is_chankan": is_chankan,
                    }
        # If nobody can naki, proceed to next draw
        if not any(d is not None for d in self.pending_decisions):
            self._chankan_tile = None
            self._next_tsumo(discarder)

    # ──────────────────────────────────────────
    # 合法性判断
    # ──────────────────────────────────────────

    def _can_ron(self, seat: int, discarder: int, tile: Tile) -> bool:
        """Furiten + agari check for ron."""
        # Permanent furiten (discarded own wait tile)
        if self.furiten[seat]:
            return False
        # Riichi permanent furiten (discarded wait tile after riichi)
        if self.riichi[seat] and self._riichi_furiten[seat]:
            return False
        # Junme (same-round) furiten: missed a winning tile this round
        if self._junme_furiten[seat]:
            return False
        # 检查加入 tile 后是否和了
        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten
        test_hand = list(self.hands[seat]) + [tile]
        counts = hand_to_counts(test_hand)
        meld_count = len(self.melds[seat])
        return calc_shanten(counts, meld_count) == -1

    def _can_chi(self, seat: int, discarder: int, tile: Tile) -> bool:
        """吃：仅上家，非字牌，且手里有配牌"""
        if (discarder + 1) % 4 != seat:
            return False
        if tile.is_honor:
            return False
        from rinshan.tile import hand_to_counts
        counts = hand_to_counts(self.hands[seat])
        num = tile.tile_id % 9
        suit = tile.tile_id - num
        for form in range(3):  # 0=高, 1=中, 2=低
            if form == 0 and num >= 2 and counts[suit+num-2] and counts[suit+num-1]:
                return True
            if form == 1 and num >= 1 and num <= 7 and counts[suit+num-1] and counts[suit+num+1]:
                return True
            if form == 2 and num <= 6 and counts[suit+num+1] and counts[suit+num+2]:
                return True
        return False

    def _can_pon(self, seat: int, tile: Tile) -> bool:
        from rinshan.tile import hand_to_counts
        counts = hand_to_counts(self.hands[seat])
        return counts[tile.tile_id] >= 2

    def _can_daiminkan(self, seat: int, tile: Tile) -> bool:
        from rinshan.tile import hand_to_counts
        counts = hand_to_counts(self.hands[seat])
        return counts[tile.tile_id] >= 3

    def _can_tsumo_agari(self, seat: int) -> bool:
        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten
        counts = hand_to_counts(self.hands[seat])
        return calc_shanten(counts, len(self.melds[seat])) == -1

    def _can_ankan(self, seat: int) -> list[int]:
        """返回可暗杠的 tile_id 列表"""
        from rinshan.tile import hand_to_counts
        counts = hand_to_counts(self.hands[seat])
        result = [i for i, c in enumerate(counts) if c >= 4]
        # 立直中只能暗杠且不改变待牌
        if self.riichi[seat]:
            result = [i for i in result if self._riichi_ankan_ok(seat, i)]
        return result

    def _riichi_ankan_ok(self, seat: int, tile_id: int) -> bool:
        """立直后暗杠是否不改变听牌（简化：检查向听数和待牌不变）"""
        from rinshan.tile import Tile as T, hand_to_counts
        from rinshan.algo.shanten import calc_shanten
        # 移除 4 张，检查向听数是否仍为 -1（即仍然听牌）
        test_counts = hand_to_counts(self.hands[seat])
        test_counts[tile_id] -= 4
        new_meld = len(self.melds[seat]) + 1
        return calc_shanten(test_counts, new_meld) == -1

    def _can_kakan(self, seat: int) -> list[int]:
        """返回可加杠的 tile_id 列表"""
        ponned = {tiles[0].tile_id for mtype, tiles in self.melds[seat] if mtype == "pon"}
        from rinshan.tile import hand_to_counts
        counts = hand_to_counts(self.hands[seat])
        return [tid for tid in ponned if counts[tid] >= 1]

    # ──────────────────────────────────────────
    # 和了得点计算（简化版，含役判断）
    # ──────────────────────────────────────────

    # ──────────────────────────────────────────
    # 内部推进
    # ──────────────────────────────────────────

    def _draw_from_yama(self) -> Tile:
        assert self.tiles_left > 0, "Yama exhausted"
        tile = self._yama.pop()
        self.tiles_left -= 1
        return tile

    def _draw_from_rinshan(self) -> Tile:
        assert self._rinshan, "Rinshan exhausted"
        tile = self._rinshan.pop()
        return tile

    def _flip_dora(self) -> None:
        if self._dora_wall:
            tile = self._dora_wall.pop()
            self.dora_indicators.append(tile)
            self._emit_dora(tile)

    def _process_reactions(self) -> None:
        """处理所有已提交的响应，推进游戏"""
        self._clear_pending_flag()
        # 收集所有响应
        reactions = {
            seat: self._reactions[seat]
            for seat in range(4)
            if self._reactions[seat] is not None
        }
        self._clear_pending()
        self._dispatch(reactions)

    def _clear_pending_flag(self) -> None:
        pass  # reserved

    def _dispatch(self, reactions: dict[int, dict]) -> None:
        """根据响应类型分派处理"""
        # 优先级：荣和 > 大明杠/碰 > 吃 > pass
        hora_actors = [
            seat for seat, r in reactions.items()
            if r.get("type") == "hora"
        ]
        if hora_actors:
            # 处理荣和（可能是多家荣，取第一个——天凤无三家荣流局，但允许双响）
            # 天凤规则：不触发三家荣流局，最多两家荣
            discarder = reactions[hora_actors[0]].get("target", -1)
            self._handle_multi_hora(hora_actors, discarder)
            return

        # 自摸和
        for seat, r in reactions.items():
            if r.get("type") == "tsumo":
                self._handle_tsumo_hora(seat)
                return

        # 打牌（含立直打牌）
        for seat, r in reactions.items():
            if r.get("type") in ("dahai", "reach"):
                self._handle_dahai_reaction(seat, r, reactions)
                return

        # 鸣牌优先级：daiminkan / pon > chi
        for seat, r in sorted(reactions.items()):
            if r.get("type") == "daiminkan":
                self._handle_daiminkan(seat, r)
                return
        for seat, r in sorted(reactions.items()):
            if r.get("type") == "pon":
                self._handle_pon(seat, r)
                return
        for seat, r in sorted(reactions.items()):
            if r.get("type") == "chi":
                self._handle_chi(seat, r)
                return

        # 暗杠 / 加杠
        for seat, r in reactions.items():
            if r.get("type") == "ankan":
                self._handle_ankan(seat, r)
                return
            if r.get("type") == "kakan":
                self._handle_kakan(seat, r)
                return

        # 九种流局
        for seat, r in reactions.items():
            if r.get("type") == "ryukyoku":
                self._handle_kyushu_ryukyoku(seat)
                return

        # 全 pass：进入下一家摸牌
        # 此时只会在打牌事件后的鸣牌询问中出现
        # _dispatch 进来的 reactions 理论上总有一个有意义的动作
        # 如果全是 pass，意味着没有人鸣牌
        self._handle_all_pass(reactions)

    def _handle_all_pass(self, reactions: dict[int, dict]) -> None:
        """All players passed. If we were in a chankan window, proceed to rinshan draw.
        Otherwise find the last discarder and advance to next tsumo."""
        if self._chankan_tile is not None:
            # Chankan: nobody robbed the kan, proceed to rinshan draw for the kan actor
            actor = None
            for ev in reversed(self._log):
                if ev["type"] == "kakan":
                    actor = ev["actor"]
                    break
            self._chankan_tile = None
            if actor is not None:
                # Draw from rinshan for the kakan actor
                tile = self._draw_from_rinshan()
                self.hands[actor].append(tile)
                # Flip dora (kakan dora flips before the rinshan draw)
                if self._pending_dora_tsumo:
                    self._pending_dora_tsumo = False
                    self._flip_dora()
                self._emit_tsumo(actor, tile)
                self._ask_discard_or_win(actor, is_tsumo=True)
            return

        # Normal pass after a dahai: find the discarder
        last_dahai = None
        for ev in reversed(self._log):
            if ev["type"] == "dahai":
                last_dahai = ev
                break
        if last_dahai is None:
            return
        discarder = last_dahai["actor"]
        self._next_tsumo(discarder)

    def _next_tsumo(self, last_discarder: int) -> None:
        """Advance to the next player's draw."""
        # Process riichi acceptance
        if self._riichi_to_accept is not None:
            seat = self._riichi_to_accept
            self._riichi_to_accept = None
            self._emit_reach_accepted(seat)
            self.scores[seat] -= 1000
            self.kyotaku += 1
            self._accepted_riichi += 1
            # Grant ippatsu to the riichi player: active until their next draw
            self._ippatsu[seat] = True
            # 四家立直流局
            if self._accepted_riichi == 4:
                self._abortive_ryukyoku("四家立直")
                return

        # 翻新宝牌（打牌后，针对大明杠/加杠）
        if self._pending_dora:
            self._pending_dora = False
            self._flip_dora()

        next_seat = (last_discarder + 1) % 4

        # Ippatsu: grant ippatsu to the riichi player AFTER reach_accepted.
        # We set it here so it's active for the *opponent's* first turn after riichi.
        # It's already been set in reach_accepted processing above if this just happened.

        # Clear junme (same-round) furiten for the player about to draw.
        # Junme furiten only lasts until the player draws their next tile.
        self._junme_furiten[next_seat] = False

        # Check牌山 is whether exhausted
        if self.tiles_left == 0:
            self._exhaustive_ryukyoku()
            return

        # 摸牌
        if self._from_rinshan:
            self._from_rinshan = False
            tile = self._draw_from_rinshan()
        else:
            tile = self._draw_from_yama()

        self.tsumo_actor = next_seat
        self.hands[next_seat].append(tile)
        self._emit_tsumo(next_seat, tile)

        # Clear junme (same-round) furiten for the drawing player: they drew,
        # so the temporary furiten from passing on a winning tile is lifted.
        self._junme_furiten[next_seat] = False

        # Consume ippatsu for the drawing player: once they draw, ippatsu is valid
        # for the upcoming discard.  We leave _ippatsu[next_seat] set here; it
        # will be consumed (read + cleared) in _full_hora_points_detail / dahai.
        # If next_seat != the riichi player, their ippatsu isn't affected.

        # 翻宝牌（kakan 后在摸牌前翻）
        if self._pending_dora_tsumo:
            self._pending_dora_tsumo = False
            self._flip_dora()

        self._ask_discard_or_win(next_seat, is_tsumo=True)

    def _handle_dahai_reaction(self, seat: int, r: dict,
                                all_reactions: dict[int, dict]) -> None:
        """处理打牌（包含立直打牌）"""
        is_reach = (r.get("type") == "reach")
        tile = Tile.from_mjai(r["pai"])
        tsumogiri = r.get("tsumogiri", False)

        # 立直宣言
        if is_reach:
            self.riichi[seat] = True
            self.riichi_junme[seat] = len(self.discards[seat])
            self._riichi_to_accept = seat
            self._emit_reach(seat)

        # 从手牌中移除
        _remove_tile(self.hands[seat], tile)
        self.discards[seat].append(tile)
        self.last_discard = tile   # Bug fix: track last discarded tile for ron

        # 流局满贯：打出非幺九牌则取消资格
        if not tile.is_yaochuhai:
            self._can_nagashi[seat] = False

        # 振听更新：如果打出的牌在自家待张里，触发振听
        self._update_furiten_on_discard(seat)

        # Ippatsu: the riichi player's ippatsu window is consumed when they discard.
        # If they didn't win (tsumo), their ippatsu expires on this discard.
        if self._ippatsu[seat]:
            self._ippatsu[seat] = False

        # Ippatsu: cancelled when the riichi player discards (right after declaration)
        # - on reach itself the ippatsu window opens; it closes on the next discard
        #   of a *different* player OR if anyone naki's.  We'll set it active after
        #   reach_accepted in _next_tsumo, and clear it here for any discard that
        #   doesn't belong to the seat that just declared riichi.
        for s in range(4):
            if s != seat and self._ippatsu[s]:
                # Another player discarded → ippatsu window survives (only naki/draw clears it)
                pass
        if not is_reach:
            # Non-riichi discard: if this seat had ippatsu from a previous reach,
            # it's already been consumed or should be cleared by a naki that follows.
            # We clear same-junme furiten for this seat since they just discarded.
            self._junme_furiten[seat] = False

        self._emit_dahai(seat, tile, tsumogiri)

        # 四风连打检查
        if self._can_four_wind:
            if self._check_four_wind_discard(seat, tile):
                self._abortive_ryukyoku("四风连打")
                return

        # 四杠散了（上一次打牌后已置标志）
        if self._check_four_kan:
            self._check_four_kan = False
            self._abortive_ryukyoku("四杠散了")
            return

        # 向鸣牌方询问
        self._ask_naki(seat, tile)

    def _update_furiten_on_discard(self, seat: int) -> None:
        """打牌后更新振听状态

        永久振听：自家打出了自己的待张
        立直振听：立直后对方打出了自家待张但没有荣和（同巡振听在同巡结束后重置）
        """
        # 此时 tile 已经追加到 discards[seat]，hands[seat] 是打牌后的手牌
        if self.furiten[seat]:
            return  # 已振听，不需要重复设置
        if self.riichi[seat]:
            # 立直中打牌就是摸切，不需要检查振听
            return
        # 永久振听：打出了自己的待张
        tile = self.discards[seat][-1]  # 刚打出的牌
        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten
        test_counts = hand_to_counts(self.hands[seat])
        test_counts[tile.tile_id] += 1  # 假设把这张牌加回来
        if calc_shanten(test_counts, len(self.melds[seat])) == -1:
            self.furiten[seat] = True

    def _check_four_wind_discard(self, seat: int, tile: Tile) -> bool:
        """四风连打检查"""
        if tile.tile_id < 27 or tile.tile_id > 30:
            self._can_four_wind = False
            return False
        if not self._is_first_round_discard(seat):
            self._can_four_wind = False
            return False
        if self._four_wind_tile is None:
            self._four_wind_tile = tile
            return False
        if self._four_wind_tile.tile_id != tile.tile_id:
            self._can_four_wind = False
            return False
        # 四家打了同一张风牌
        return seat == (self.oya + 3) % 4  # 最后一家（北家）打出时触发

    def _is_first_round_discard(self, seat: int) -> bool:
        # Called AFTER the tile has been appended to discards[seat],
        # so the first discard means len == 1.
        return len(self.discards[seat]) == 1

    def _handle_tsumo_hora(self, seat: int) -> None:
        """处理自摸和"""
        win_tile = self.hands[seat][-1] if self.hands[seat] else Tile(0)
        # 里宝牌
        ura_count = min(sum(1 for r in self.riichi if r),
                        max(0, 5 - len(self._dora_wall)))
        ura = list(reversed(self._ura_wall[:ura_count]))

        result = self._full_hora_points_detail(seat, seat, win_tile, ura)
        deltas = [0] * 4
        is_oya = (seat == self.oya)
        if is_oya:
            for s in range(4):
                if s != seat:
                    deltas[s] = -result.tsumo_ko - self.honba * 100
            deltas[seat] = result.tsumo_ko * 3 + self.kyotaku * 1000 + self.honba * 300
        else:
            for s in range(4):
                if s == seat:
                    deltas[s] = (result.tsumo_ko * 2 + result.tsumo_oya
                                 + self.kyotaku * 1000 + self.honba * 300)
                elif s == self.oya:
                    deltas[s] = -result.tsumo_oya - self.honba * 100
                else:
                    deltas[s] = -result.tsumo_ko - self.honba * 100

        self._emit_hora(seat, seat, deltas, ura)
        self._end_kyoku(deltas, has_hora=True,
                        can_renchan=(seat == self.oya))

    def _handle_multi_hora(self, actors: list[int], discarder: int) -> None:
        """处理荣和（单家或双家）"""
        win_tile = self.last_discard or Tile(0)
        # 里宝牌
        ura_count = min(sum(1 for r in self.riichi if r),
                        max(0, 5 - len(self._dora_wall)))
        ura = list(reversed(self._ura_wall[:ura_count]))

        kyotaku_given = False
        honba_given   = False
        for actor in actors:
            result = self._full_hora_points_detail(actor, discarder, win_tile, ura)
            deltas = [0] * 4
            if not kyotaku_given:
                deltas[actor] += self.kyotaku * 1000
                kyotaku_given = True
            if not honba_given:
                deltas[actor]    += self.honba * 300
                deltas[discarder]-= self.honba * 300
                honba_given = True
            deltas[discarder] -= result.ron
            deltas[actor]     += result.ron
            self._emit_hora(actor, discarder, deltas, ura)

        # 汇总 hora 事件的 deltas
        total_deltas = [0] * 4
        hora_events = [ev for ev in self._log if ev["type"] == "hora"]
        for ev in hora_events[-len(actors):]:
            for i in range(4):
                total_deltas[i] += ev["deltas"][i]

        can_renchan = any(a == self.oya for a in actors)
        self._end_kyoku(total_deltas, has_hora=True, can_renchan=can_renchan)

    def _full_hora_points_detail(
        self, actor: int, target: int, win_tile: Tile,
        ura_markers: list[Tile]
    ):
        """调用 HoraCalculator 返回 HoraResult"""
        try:
            from rinshan.engine.hora_calc import HoraCalculator
        except ImportError:
            from rinshan.engine.hora_calc import HoraCalculator

        calc = HoraCalculator()
        is_tsumo = (actor == target)

        # 手牌（不含和了牌）
        # B5 fix: 按 tile_id 移除和了牌而不是盲目取 [-1]，避免赤宝牌原因导致删错牌
        hand = list(self.hands[actor])
        if is_tsumo:
            removed = False
            for idx in range(len(hand) - 1, -1, -1):
                if hand[idx].tile_id == win_tile.tile_id and hand[idx].is_aka == win_tile.is_aka:
                    hand.pop(idx)
                    removed = True
                    break
            if not removed:
                for idx in range(len(hand) - 1, -1, -1):
                    if hand[idx].tile_id == win_tile.tile_id:
                        hand.pop(idx)
                        break

        # 副露
        melds = list(self.melds[actor])
        is_menzen = len(melds) == 0

        # 场况判断
        is_riichi  = self.riichi[actor]
        # Ippatsu: consume it now (read then clear), valid only for this win
        is_ippatsu = self._ippatsu[actor] if is_riichi else False
        if is_ippatsu:
            self._ippatsu[actor] = False  # consumed by the win
        is_rinshan = self._from_rinshan  # 岭上开花
        # Chankan: True when actor wins by robbing a kakan
        is_chankan = (self._chankan_tile is not None and not is_tsumo)
        is_haitei  = (is_tsumo and self.tiles_left == 0)
        is_houtei  = (not is_tsumo and self.tiles_left == 0)
        is_tenhou  = (actor == self.oya and
                      len(self.discards[actor]) == 0 and
                      all(len(self.discards[s]) == 0 for s in range(4))
                      and is_tsumo)
        is_chiihou = (actor != self.oya and
                      len(self.discards[actor]) == 0 and
                      not any(len(self.melds[s]) > 0 for s in range(4))
                      and is_tsumo)

        # 自风：actor 相对于庄家的座位差
        player_wind = (actor - self.oya) % 4
        round_wind  = self.kyoku // 4   # 0=东场 1=南场

        result = calc.calc(
            hand=hand, win_tile=win_tile, melds=melds,
            is_tsumo=is_tsumo, is_riichi=is_riichi,
            is_ippatsu=is_ippatsu, is_rinshan=is_rinshan,
            is_chankan=is_chankan, is_haitei=is_haitei,
            is_houtei=is_houtei, is_tenhou=is_tenhou,
            is_chiihou=is_chiihou,
            dora_indicators=list(self.dora_indicators),
            ura_indicators=ura_markers,
            player_wind=player_wind,
            round_wind=round_wind,
        )

        if result.error is not None:
            # 无役 fallback（可能是 random agent 的无役和了）：给 1000 点
            from rinshan.engine.hora_calc import HoraResult
            result = HoraResult(
                han=1, fu=30,
                ron=1000, tsumo_ko=300, tsumo_oya=500,
                error=None,
            )
        return result

    def _handle_chi(self, actor: int, r: dict) -> None:
        discarder = r.get("discarder", r.get("target"))
        pai = Tile.from_mjai(r["pai"])
        consumed = [Tile.from_mjai(t) for t in r["consumed"]]
        for t in consumed:
            _remove_tile(self.hands[actor], t)
        self.melds[actor].append(("chi", [pai] + consumed))
        self._can_nagashi[discarder] = False
        self._can_four_wind = False
        # Any naki cancels all ippatsu
        self._ippatsu = [False] * 4
        self._emit_chi(actor, discarder, pai, consumed)
        self._ask_discard_or_win(actor, is_tsumo=False)

    def _handle_pon(self, actor: int, r: dict) -> None:
        discarder = r.get("discarder", r.get("target"))
        pai = Tile.from_mjai(r["pai"])
        consumed = [Tile.from_mjai(t) for t in r["consumed"]]
        for t in consumed:
            _remove_tile(self.hands[actor], t)
        self.melds[actor].append(("pon", [pai] + consumed))
        self._can_nagashi[discarder] = False
        self._can_four_wind = False
        self._update_paos(actor, discarder, pai, "pon")
        # Any naki cancels all ippatsu
        self._ippatsu = [False] * 4
        self._emit_pon(actor, discarder, pai, consumed)
        self._ask_discard_or_win(actor, is_tsumo=False)

    def _handle_daiminkan(self, actor: int, r: dict) -> None:
        discarder = r.get("discarder", r.get("target"))
        pai = Tile.from_mjai(r["pai"])
        consumed = [Tile.from_mjai(t) for t in r["consumed"]]
        for t in consumed:
            _remove_tile(self.hands[actor], t)
        self.melds[actor].append(("daiminkan", [pai] + consumed))
        self._can_nagashi[discarder] = False
        self._can_four_wind = False
        self._pending_dora = True  # 打牌后翻宝牌
        self.kans += 1
        self._update_paos(actor, discarder, pai, "daiminkan")
        # Any naki cancels all ippatsu
        self._ippatsu = [False] * 4
        self._emit_daiminkan(actor, discarder, pai, consumed)
        # Bug fix: draw from rinshan immediately (same as ankan/kakan),
        # do NOT set _from_rinshan=True and defer the draw to _next_tsumo
        # (_next_tsumo uses (last_discarder+1)%4 for next_seat, which would
        #  give the WRONG seat the rinshan tile after the actor discards)
        tile = self._draw_from_rinshan()
        self.hands[actor].append(tile)
        self._emit_tsumo(actor, tile)
        self._ask_discard_or_win(actor, is_tsumo=True)

    def _handle_ankan(self, actor: int, r: dict) -> None:
        consumed = [Tile.from_mjai(t) for t in r["consumed"]]
        tile_id = consumed[0].tile_id
        # 从手牌移除 4 张
        for t in consumed:
            _remove_tile(self.hands[actor], t)
        self.melds[actor].append(("ankan", consumed))
        self.kans += 1
        # 暗杠立即翻宝牌
        self._flip_dora()
        self._emit_ankan(actor, consumed)
        # 四杠散了判断
        if self.kans == 4 and not any(
            sum(1 for mtype, _ in self.melds[s] if "kan" in mtype) == 4
            for s in range(4)
        ):
            self._check_four_kan = True
        # 岭上摸牌
        tile = self._draw_from_rinshan()
        self.hands[actor].append(tile)
        self._emit_tsumo(actor, tile)
        self._ask_discard_or_win(actor, is_tsumo=True)

    def _handle_kakan(self, actor: int, r: dict) -> None:
        pai = Tile.from_mjai(r["pai"])
        # 从手牌移除 1 张
        _remove_tile(self.hands[actor], pai)
        # 找到对应的碰，转为加杠
        for i, (mtype, tiles) in enumerate(self.melds[actor]):
            if mtype == "pon" and tiles[0].tile_id == pai.tile_id:
                self.melds[actor][i] = ("kakan", tiles + [pai])
                break
        self.kans += 1
        self._pending_dora_tsumo = True  # kakan 宝牌在岭上摸牌前翻
        self._emit_kakan(actor, pai)
        # 四杠散了判断（同暗杠）
        if self.kans == 4 and not any(
            sum(1 for mtype, _ in self.melds[s] if "kan" in mtype) == 4
            for s in range(4)
        ):
            self._check_four_kan = True

        # Chankan (抢杠): ask all other players if they can ron this tile.
        # This is like a discard but only RON is allowed (no chi/pon/daiminkan).
        # Set _chankan_tile so _full_hora_points_detail picks up is_chankan=True.
        self._ask_naki(actor, pai, is_chankan=True)

        # If nobody robbed the kan, _next_tsumo won't be called by _ask_naki
        # (it only calls _next_tsumo when nobody has a pending decision).
        # After chankan reactions are resolved, the actor draws from rinshan.
        # We set up the rinshan draw in _dispatch after all chankan passes.

    def _handle_kyushu_ryukyoku(self, seat: int) -> None:
        """九种九牌流局"""
        self._abortive_ryukyoku("九种九牌")

    def _update_paos(self, actor: int, target: int,
                     pai: Tile, mtype: str) -> None:
        """包牌判断（三元牌/四风牌）"""
        if not pai.is_honor:
            return
        # 大三元：已有白发，再碰中 → target 为包牌者
        jihais = [t.tile_id for mtype2, tiles in self.melds[actor]
                  for t in tiles if t.tile_id >= 31]  # 31=白 32=發 33=中
        wind_ji = [t.tile_id for mtype2, tiles in self.melds[actor]
                   for t in tiles if 27 <= t.tile_id <= 30]
        if len(set(jihais)) == 3:
            self._paos[actor] = target
        if len(set(wind_ji)) == 4:
            self._paos[actor] = target

    def _abortive_ryukyoku(self, reason: str) -> None:
        deltas = [0] * 4
        tenpai = [False] * 4
        self._emit_ryukyoku(reason, deltas, tenpai)
        self._end_kyoku(deltas, has_hora=False, can_renchan=True,
                        is_abortive=True)

    def _exhaustive_ryukyoku(self) -> None:
        """荒牌流局"""
        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten

        def _is_tenpai(seat: int) -> bool:
            hand = self.hands[seat]
            meld_count = len(self.melds[seat])
            # 手牌可能是 14 张（当前摸牌者还没打牌）或 13 张
            # 只要 shanten <= 0（听牌或已和）就算听牌
            counts = hand_to_counts(hand)
            sht = calc_shanten(counts, meld_count)
            if sht <= 0:
                return True
            # 如果 14 张，尝试去掉任意一张后是否 shanten==0
            if sum(counts) == 14:
                for i in range(34):
                    if counts[i] > 0:
                        counts[i] -= 1
                        sht2 = calc_shanten(counts, meld_count)
                        counts[i] += 1
                        if sht2 == 0:
                            return True
            return False

        tenpai = [_is_tenpai(s) for s in range(4)]

        # 流局满贯（流局时只剩幺九牌）
        has_nagashi = [self._can_nagashi[s] for s in range(4)]
        deltas = [0] * 4

        if any(has_nagashi):
            for s in range(4):
                if has_nagashi[s]:
                    if s == self.oya:
                        # 庄家流满：各家支付 4000，庄家收 12000（自己不参与支付）
                        for t in range(4):
                            if t != s:
                                deltas[t] -= 4000
                        deltas[s] += 12000
                    else:
                        # 非庄家流满：非庄各家支付 2000，庄家支付 4000，winner 收 8000
                        for t in range(4):
                            if t != s:
                                deltas[t] -= 2000
                        deltas[self.oya] -= 2000  # 庄家额外多支付 2000
                        deltas[s] += 8000
        else:
            n_tenpai = sum(tenpai)
            if 0 < n_tenpai < 4:
                pay_map = {1: (3000, 1000), 2: (1500, 1500), 3: (1000, 3000)}
                plus, minus = pay_map[n_tenpai]
                for s in range(4):
                    deltas[s] = plus if tenpai[s] else -minus

        can_renchan = tenpai[self.oya]
        self._emit_ryukyoku("荒牌", deltas, tenpai)
        self._end_kyoku(deltas, has_hora=False, can_renchan=can_renchan)

    def _end_kyoku(
        self,
        deltas: list[int],
        has_hora: bool,
        can_renchan: bool,
        is_abortive: bool = False,
    ) -> None:
        # 供托归并：立直和了时供托全部归赢家
        new_scores = [self.scores[i] + deltas[i] for i in range(4)]
        self._emit_end_kyoku()

        # 供托棒由外部的 Game 在下一局分配
        self.result = KyokuResult(
            kyoku     = self.kyoku,
            honba     = self.honba,
            can_renchan             = can_renchan,
            has_hora                = has_hora,
            has_abortive_ryukyoku   = is_abortive,
            kyotaku_left            = self.kyotaku,
            deltas                  = list(deltas),
            scores_after            = new_scores,
        )
        self._done = True


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _remove_tile(hand: list[Tile], tile: Tile) -> None:
    """优先移除匹配赤标记的牌"""
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id and t.is_aka == tile.is_aka:
            hand.pop(i)
            return
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id:
            hand.pop(i)
            return
    raise ValueError(f"Tile {tile} not in hand: {hand}")
