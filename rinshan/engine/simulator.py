"""
MjaiSimulator — 消费 mjai 格式 JSON 事件流，生成 Annotation 序列

流程：
  事件流（list of dict）
    → 逐局维护 GameState
    → 每个决策点生成 Annotation（未填 GRP 奖励）
    → 局结束后回填 delta score / rank 等结果标注

兼容天凤 mjai 格式（.json.gz，每行一个事件或完整局列表）。
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Iterator

from rinshan.tile import Tile, hand_to_counts
from rinshan.algo.shanten import calc_shanten
from rinshan.constants import (
    PROG_DISCARD_BASE, PROG_DISCARD_TSUMOGIRI_BASE,
    PROG_DRAW_BASE, PROG_RIICHI_BASE,
    PROG_CHI_BASE, PROG_PON_BASE, PROG_DAIMINKAN_BASE,
    PROG_ANKAN_BASE, PROG_KAKAN_BASE, PROG_NEWDORA_BASE,
)
from .state   import GameState, PlayerView, ActionCandidate
from .action  import (
    Action, ActionType,
    encode_action, chi_type_to_idx,
)
from rinshan.data.annotation import Annotation, AuxTargets


class MjaiSimulator:
    """
    逐局解析 mjai 事件流，输出 Annotation 列表

    Usage:
        sim = MjaiSimulator()
        for ann in sim.parse_game(events, game_id="xxx"):
            process(ann)
    """

    def parse_game(
        self,
        events: list[dict],
        game_id: str = "unknown",
    ) -> list[Annotation]:
        """
        解析一局完整游戏的事件列表，返回所有决策点的 Annotation

        events: mjai 格式事件列表，形如：
          [{"type":"start_game", ...},
           {"type":"start_kyoku", ...},
           {"type":"tsumo", ...},
           {"type":"dahai", ...},
           ...]
        """
        annotations: list[Annotation] = []
        state = GameState()
        kyoku_annotations: list[Annotation] = []   # 本局的标注，局结束后回填

        for i, event in enumerate(events):
            etype = event.get("type", "")

            if etype == "start_game":
                state = GameState()

            elif etype == "start_kyoku":
                state, kyoku_annotations = self._handle_start_kyoku(
                    event, state, kyoku_annotations, annotations
                )

            elif etype == "tsumo":
                self._handle_tsumo(event, state)

            elif etype == "dahai":
                new_anns = self._handle_dahai(event, state, game_id)
                kyoku_annotations.extend(new_anns)

                # Look ahead: generate naki/ron/pass Annotations for all eligible players
                reaction_anns = self._handle_post_discard_reactions(
                    event, events[i + 1:], state, game_id
                )
                kyoku_annotations.extend(reaction_anns)

            elif etype in ("chi", "pon", "daiminkan"):
                # Annotations were already generated in _handle_post_discard_reactions.
                # Only update game state here.
                self._update_state_naki(event, state)

            elif etype in ("ankan", "kakan"):
                new_anns = self._handle_kan(event, state, game_id)
                kyoku_annotations.extend(new_anns)

            elif etype == "reach":
                self._handle_riichi(event, state)

            elif etype == "reach_accepted":
                seat = event.get("actor", 0)
                state.riichi_accepted[seat] = True
                state.in_riichi[seat] = True
                state.kyotaku += 1

            elif etype == "dora":
                tile = Tile.from_mjai(event["dora_marker"])
                state.dora_indicators.append(tile)
                state.progression.append(PROG_NEWDORA_BASE + tile.tile_id)

            elif etype == "hora":
                # Generate TSUMO annotation if self-draw; RON already generated post-discard.
                new_anns = self._handle_hora(event, state, game_id)
                kyoku_annotations.extend(new_anns)
                self._handle_end_kyoku(event, state, kyoku_annotations, annotations)
                kyoku_annotations = []

            elif etype in ("ryukyoku", "ryuukyoku"):
                # Generate kyushu annotation if applicable.
                new_anns = self._handle_ryukyoku(event, state, game_id)
                kyoku_annotations.extend(new_anns)
                self._handle_end_kyoku(event, state, kyoku_annotations, annotations)
                kyoku_annotations = []

            elif etype == "end_kyoku":
                # 整场对局结算：更新最终分数，回填 final_delta_score / final_rank
                self._handle_end_game(event, state, annotations)

            elif etype == "end_game":
                break

        return annotations

    # ─────────────────────────────────────────
    # 事件处理器
    # ─────────────────────────────────────────

    def _handle_start_kyoku(
        self,
        event: dict,
        state: GameState,
        kyoku_anns: list[Annotation],
        all_anns: list[Annotation],
    ):
        """新局开始：重置局面，注意保留分数"""
        # 上局未正常结束就收到新局（数据异常）：丢弃未回填的标注，不写入 all_anns
        # 注：正常情况下 kyoku_anns 应已在 _handle_end_kyoku 里被清空
        if kyoku_anns:
            import logging
            logging.getLogger(__name__).warning(
                f"start_kyoku with {len(kyoku_anns)} un-finalized annotations, discarding"
            )

        # 解析新局信息
        bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
        bakaze = bakaze_map.get(event.get("bakaze", "E"), 0)
        kyoku  = event.get("kyoku", 1)
        honba  = event.get("honba", 0)
        kyotaku= event.get("kyotaku", 0)
        scores = event.get("scores", [25000]*4)
        dealer = event.get("oya", 0)

        # 手牌分发
        haipai = event.get("tehais", [[], [], [], []])
        hands  = []
        for h in haipai:
            hand = []
            for t in h:
                if t == "?":
                    continue   # 推理场景对手手牌不可见；复盘牌谱不会出现 '?'
                hand.append(Tile.from_mjai(t))
            hands.append(hand)

        dora_indicators = [Tile.from_mjai(event["dora_marker"])] if "dora_marker" in event else []

        new_state = GameState(
            round_wind      = bakaze,
            round_num       = kyoku,
            honba           = honba,
            kyotaku         = kyotaku,
            dealer          = dealer,
            scores          = list(scores),
            tiles_left      = 70,
            dora_indicators = dora_indicators,
            hands           = [list(h) for h in hands],
            discards        = [[] for _ in range(4)],
            melds           = [[] for _ in range(4)],
            riichi_declared = [False]*4,
            riichi_accepted = [False]*4,
            in_riichi       = [False]*4,
            current_player  = dealer,
            progression     = [],
        )
        return new_state, []

    def _handle_tsumo(self, event: dict, state: GameState):
        seat = event.get("actor", state.current_player)
        tile_str = event.get("pai", "?")
        if tile_str == "?":
            # 推理场景对手摸牌不可见；复盘牌谱不会出现 '?'
            state.tiles_left -= 1
            state.progression.append(PROG_DRAW_BASE + seat)
            return
        tile = Tile.from_mjai(tile_str)
        state.hands[seat].append(tile)
        state.tiles_left -= 1
        state.last_draw = tile
        state.current_player = seat
        state.progression.append(PROG_DRAW_BASE + seat)

    def _handle_dahai(
        self, event: dict, state: GameState, game_id: str
    ) -> list[Annotation]:
        """打牌事件 → 生成打牌决策点标注"""
        seat      = event.get("actor", 0)
        tile_str  = event.get("pai", "1z")
        tile      = Tile.from_mjai(tile_str)
        tsumogiri = event.get("tsumogiri", False)

        # ── 在移除手牌之前 生成标注 ──────────────────
        # 此时手牌 = 摸牌后的完整手牌（包含待打出的 tile）
        anns = []
        cans  = self._compute_discard_candidates(state, seat)
        cands = self._build_candidate_tokens(cans, ActionType.DISCARD)
        if cands:
            # 判断是否为立直宣言时的打牌
            if state.riichi_declared[seat] and state.riichi_discard_idx[seat] == len(state.discards[seat]):
                action = Action(ActionType.RIICHI)
            else:
                action = Action(ActionType.DISCARD, tile=tile)
            
            chosen = self._find_action_idx(cands, action)
            view   = state.player_view(seat)
            ann    = self._make_annotation(game_id, seat, view, state, cands, chosen)
            anns.append(ann)

        # ── 更新状态 ──────────────────────────────────
        _remove_tile(state.hands[seat], tile)
        state.discards[seat].append(tile)
        state.last_discard = tile
        state.last_draw    = None

        # 振听更新（永久振听）：
        # 打出了自己的待张之一 → 振听
        # 注意：此时 tile 已从 hands 移除，hands 是打牌后的 13 张（门清）或副露后的更少
        # 正确判断：把打出的牌加回 13 张手牌 → 向听数 == -1 说明打出的牌是待张
        if not state.furiten[seat] and not state.in_riichi[seat]:
            test_counts = hand_to_counts(state.hands[seat])
            test_counts[tile.tile_id] += 1
            if calc_shanten(test_counts, len(state.melds[seat])) == -1:
                # 打出的牌是自己的待张 → 永久振听
                state.furiten[seat] = True

        tile_idx = tile.tile_id if not tile.is_aka else {4: 34, 13: 35, 22: 36}[tile.tile_id]
        discard_base = PROG_DISCARD_TSUMOGIRI_BASE if tsumogiri else PROG_DISCARD_BASE
        prog_tok = discard_base + seat * 37 + tile_idx
        state.progression.append(prog_tok)

        return anns

    def _handle_post_discard_reactions(
        self, discard_event: dict, next_events: list[dict], state: GameState, game_id: str
    ) -> list[Annotation]:
        """
        After a discard, look ahead to see who reacted (naki or ron).
        Generate Annotation for everyone eligible.
        If they reacted, their chosen action is what they did.
        If they didn't (or weren't the ones who won in case of multiple eligible), they PASSED.
        """
        target = discard_event.get("actor", 0)
        pai = Tile.from_mjai(discard_event["pai"])
        
        # What did players do?
        # A player could:
        # - Hora (ron)
        # - Chi, Pon, Daiminkan
        # - Nothing (Pass)
        
        # Look ahead until we hit an event that resolves the reaction window
        # (which is when the next turn begins: tsumo, or the state advances via naki/hora)
        reactions = {}
        consumed_map = {}
        for ev in next_events:
            etype = ev.get("type", "")
            if etype == "hora":
                actor = ev.get("actor", 0)
                # target for hora should match discarder
                ev_target = ev.get("target", actor)
                if ev_target == target and actor != target:
                    reactions[actor] = ActionType.RON
            elif etype in ("chi", "pon", "daiminkan"):
                actor = ev.get("actor", 0)
                reactions[actor] = {"chi": ActionType.CHI, "pon": ActionType.PON, "daiminkan": ActionType.DAIMINKAN}[etype]
                consumed_map[actor] = [Tile.from_mjai(t) for t in ev.get("consumed", [])]
                break # Naki ends the reaction window (only one naki allowed)
            else:
                # Any other event (e.g. tsumo, ryukyoku, start_kyoku) means no more reactions to this discard
                break

        anns: list[Annotation] = []
        for seat in range(4):
            if seat == target:
                continue
            
            cands = self._compute_naki_candidates(state, seat, target, pai)
            if not cands:
                continue
            
            # They were eligible, what did they do?
            rtype = reactions.get(seat, ActionType.PASS)
            
            if rtype == ActionType.RON:
                actual_action = Action(ActionType.RON, actor=seat)
            elif rtype in (ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN):
                actual_action = Action(
                    type=rtype,
                    tile=pai,
                    consumed=consumed_map[seat],
                    actor=seat,
                    target=target,
                )
            else:
                actual_action = Action(ActionType.PASS, actor=seat)

            chosen = self._find_action_idx(cands, actual_action)
            view = state.player_view(seat)
            ann  = self._make_annotation(game_id, seat, view, state, cands, chosen)
            anns.append(ann)
            
        return anns

    def _update_state_naki(self, event: dict, state: GameState):
        """Update state for naki (chi/pon/daiminkan) - annotations are handled post-discard"""
        etype    = event["type"]
        actor    = event.get("actor", 0)
        target   = event.get("target", (actor - 1) % 4)
        pai      = Tile.from_mjai(event["pai"])
        consumed = [Tile.from_mjai(t) for t in event.get("consumed", [])]
        
        if etype == "chi":
            t1, t2 = sorted(consumed, key=lambda t: t.tile_id)
            suit = t1.tile_id // 9
            low  = min(t1.tile_id, t2.tile_id) % 9 + 1
            t_num = pai.number
            if   t_num == low + 2: form = 0
            elif t_num == low + 1: form = 1
            else:                  form = 2
            from rinshan.engine.action import chi_type_to_idx
            prog_tok = PROG_CHI_BASE + chi_type_to_idx(suit, low, form)
        elif etype == "pon":
            prog_tok = PROG_PON_BASE + actor * 34 + pai.tile_id
        else: # daiminkan
            prog_tok = PROG_DAIMINKAN_BASE + actor * 34 + pai.tile_id

        for t in consumed:
            _remove_tile(state.hands[actor], t)
        state.melds[actor].append((etype, [pai] + consumed))
        state.progression.append(prog_tok)
        state.current_player = actor
        state.last_draw = None

    def _handle_hora(self, event: dict, state: GameState, game_id: str) -> list[Annotation]:
        """Tsumo hora generates an annotation. Ron hora is already generated in post_discard_reactions."""
        actor = event.get("actor", 0)
        target = event.get("target", actor)
        anns = []
        if actor == target:
            # Tsumo
            cans = self._compute_discard_candidates(state, actor)
            cands = self._build_candidate_tokens(cans, ActionType.DISCARD)  # TSUMO is one of the candidates here
            if cands:
                chosen = self._find_action_idx(cands, Action(ActionType.TSUMO, actor=actor))
                view = state.player_view(actor)
                ann = self._make_annotation(game_id, actor, view, state, cands, chosen)
                anns.append(ann)
        return anns

    def _handle_ryukyoku(self, event: dict, state: GameState, game_id: str) -> list[Annotation]:
        """Ryukyoku generates an annotation if it's kyushu kyuhai (turn action)"""
        anns = []
        if event.get("reason") == "kyushukyuhai" or event.get("reason") == "九種九牌":
            actor = event.get("actor", state.current_player)
            cans = self._compute_discard_candidates(state, actor)
            # Add ryukyoku candidate
            from rinshan.constants import RYUKYOKU_TOKEN
            cands = self._build_candidate_tokens(cans, ActionType.DISCARD)
            cands.append(RYUKYOKU_TOKEN)
            chosen = self._find_action_idx(cands, Action(ActionType.RYUKYOKU, actor=actor))
            view = state.player_view(actor)
            ann = self._make_annotation(game_id, actor, view, state, cands, chosen)
            anns.append(ann)
        return anns

    def _handle_kan(self, event: dict, state: GameState, game_id: str) -> list[Annotation]:
        """暗杠/加杠事件，在更新状态前生成决策点"""
        etype    = event["type"]
        actor    = event.get("actor", 0)
        consumed = [Tile.from_mjai(t) for t in event.get("consumed", [])]

        anns: list[Annotation] = []

        if etype == "ankan":
            # 暗杠候选：手里所有有4张的牌
            cands = self._compute_ankan_candidates(state, actor)
            if cands:
                tile = consumed[0] if consumed else None
                action = Action(ActionType.ANKAN, tile=tile, consumed=consumed, actor=actor)
                chosen = self._find_action_idx(cands, action)
                view   = state.player_view(actor)
                anns.append(self._make_annotation(game_id, actor, view, state, cands, chosen))

            for t in consumed:
                _remove_tile(state.hands[actor], t)
            state.melds[actor].append(("ankan", consumed))
            state.progression.append(PROG_ANKAN_BASE + actor * 34 + consumed[0].tile_id)

        else:  # kakan
            # 加杠候选：副露里有碰，且手里还有同种牌
            cands = self._compute_kakan_candidates(state, actor)
            tile  = Tile.from_mjai(event["pai"])
            if cands:
                action = Action(ActionType.KAKAN, tile=tile, actor=actor)
                chosen = self._find_action_idx(cands, action)
                view   = state.player_view(actor)
                anns.append(self._make_annotation(game_id, actor, view, state, cands, chosen))

            _remove_tile(state.hands[actor], tile)
            # P3 同款 fix: 把 melds 里对应的 pon 升级为 kakan
            for i, (mtype, tiles) in enumerate(state.melds[actor]):
                if mtype == "pon" and tiles[0].tile_id == tile.tile_id:
                    state.melds[actor][i] = ("kakan", tiles + [tile])
                    break
            state.progression.append(PROG_KAKAN_BASE + actor * 34 + tile.tile_id)

        return anns

    def _handle_riichi(self, event: dict, state: GameState):
        seat = event.get("actor", 0)
        state.riichi_declared[seat] = True
        # 记录立直宣言时已有的弃牌数，下一张打出的牌就是立直牌
        state.riichi_discard_idx[seat] = len(state.discards[seat])
        state.progression.append(PROG_RIICHI_BASE + seat)

    def _handle_end_game(
        self,
        event: dict,
        state: GameState,
        all_anns: list[Annotation],
    ):
        """
        end_kyoku 事件（对局最终结算）：
        用 final_scores 回填所有标注的 final_delta_score 和 final_rank。
        天凤格式里 final_scores 是绝对分，需要与对局开始分数做差。
        """
        final_scores = event.get("final_scores")
        if not final_scores or len(final_scores) < 4:
            return

        # 计算最终排名
        final_ranks = _scores_to_ranks(final_scores)

        # 回填所有标注（game 内按 player_id 定位起始分数）
        # 注意：起始分数在 start_game 时都是 25000，用 final_scores - start_scores
        # 由于 start_scores 已经在每局开始时记录，这里只需要知道初始分
        # 简化处理：final_delta = final_scores[seat] - state.scores[seat]（当前分）
        # 但 state.scores 已经是最后一局结束后的分数，和 final_scores 相同
        # 所以直接用 final_scores 算排名，然后找出每个玩家的最终分变化
        # （初始分默认 25000，end_game 时已经历过多局累加）
        for ann in all_anns:
            seat = ann.player_id
            ann.final_rank = final_ranks[seat]
            # final_delta_score：终局分 - 初始分（25000）
            ann.final_delta_score = final_scores[seat] - 25000

    def _handle_end_kyoku(
        self,
        event: dict,
        state: GameState,
        kyoku_anns: list[Annotation],
        all_anns: list[Annotation],
    ):
        """局结束：回填结果标注"""
        etype = event["type"]

        # 解析得分变化
        deltas = event.get("deltas", [0]*4)
        # 解析最终排名
        new_scores = [state.scores[i] + deltas[i] for i in range(4)]
        ranks = _scores_to_ranks(new_scores)

        # 回填每条标注的 round_delta_score、final_rank
        for ann in kyoku_anns:
            seat = ann.player_id
            ann.round_delta_score = deltas[seat]
            ann.final_rank        = ranks[seat]

        all_anns.extend(kyoku_anns)

        # 更新分数
        for i in range(4):
            state.scores[i] = new_scores[i]
        # B9 fix: 只有和了才清空供托，流局时供托保留到下一局
        if etype == "hora" and event.get("deltas"):
            state.kyotaku = 0
        # 流局时供托棒数 +1（立直棒仍保留，下局继续累积）
        # 注意：kyotaku 由 reach_accepted 事件维护，这里不需要额外处理

    # ─────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────

    def _compute_ankan_candidates(self, state: GameState, seat: int) -> list[int]:
        """摸牌后可以暗杠的候选 token 列表（含 PASS）"""
        from rinshan.constants import ANKAN_OFFSET, PASS_TOKEN
        counts = [0] * 34
        for t in state.hands[seat]:
            counts[t.tile_id] += 1
        tokens = [ANKAN_OFFSET + i for i, c in enumerate(counts) if c >= 4]
        if not tokens:
            return []
        tokens.append(PASS_TOKEN)
        return tokens

    def _compute_kakan_candidates(self, state: GameState, seat: int) -> list[int]:
        """摸牌后可以加杠的候选 token 列表（含 PASS）"""
        from rinshan.constants import KAKAN_OFFSET, PASS_TOKEN
        ponned = set()
        for meld_type, tiles in state.melds[seat]:
            if meld_type == "pon":
                ponned.add(tiles[0].tile_id)
        counts = [0] * 34
        for t in state.hands[seat]:
            counts[t.tile_id] += 1
        tokens = [KAKAN_OFFSET + tid for tid in ponned if counts[tid] >= 1]
        if not tokens:
            return []
        tokens.append(PASS_TOKEN)
        return tokens

    def _compute_naki_candidates(
        self,
        state: GameState,
        seat: int,
        target: int,
        pai: "Tile",
    ) -> list[int]:
        """
        计算 seat 对 target 打出的 pai 能执行的鸣牌候选 token 列表。
        始终包含 PASS_TOKEN（选择不鸣）。
        若无任何鸣牌机会（只剩 PASS），返回空列表——表示不生成决策点。
        立直中的玩家只能荣和或 pass，不能吃/碰/杠。
        """
        from rinshan.constants import (
            CHI_OFFSET, PON_OFFSET, DAIMINKAN_OFFSET, PASS_TOKEN,
            RON_AGARI_TOKEN,
        )
        from rinshan.engine.action import chi_type_to_idx

        hand = state.hands[seat]
        counts = [0] * 34
        for t in hand:
            counts[t.tile_id] += 1

        tokens: list[int] = []

        # 立直中：只能荣和，不能吃/碰/杠
        if state.in_riichi[seat]:
            if not state.furiten[seat]:
                test_counts = list(counts)
                test_counts[pai.tile_id] += 1
                sht = calc_shanten(test_counts, len(state.melds[seat]))
                if sht == -1:
                    tokens.append(RON_AGARI_TOKEN)
            if not tokens:
                return []  # 立直中不能荣和，不生成决策点
            tokens.append(PASS_TOKEN)
            return tokens

        # ── 碰（任意方向均可）────────────────────────
        if counts[pai.tile_id] >= 2:
            tokens.append(PON_OFFSET + pai.tile_id)

        # ── 大明杠（手里有3张）───────────────────────
        if counts[pai.tile_id] >= 3:
            tokens.append(DAIMINKAN_OFFSET + pai.tile_id)

        # ── 吃（仅上家，且不是字牌）─────────────────
        is_upper = (target == (seat - 1) % 4)
        if is_upper and not pai.is_honor:
            suit  = pai.tile_id // 9
            num   = pai.tile_id % 9  # 0-based

            # form 0: 吃对象是搭子最高张 → 手里需要 num-2, num-1 (0-based)
            if num >= 2 and counts[suit*9 + num - 2] > 0 and counts[suit*9 + num - 1] > 0:
                tokens.append(CHI_OFFSET + chi_type_to_idx(suit, num - 1, 0))  # low=num-1 (1-based)

            # form 1: 吃对象是搭子中间张 → 手里需要 num-1, num+1
            if num >= 1 and num <= 7 and counts[suit*9 + num - 1] > 0 and counts[suit*9 + num + 1] > 0:
                tokens.append(CHI_OFFSET + chi_type_to_idx(suit, num, 1))      # low=num (1-based)

            # form 2: 吃对象是搭子最低张 → 手里需要 num+1, num+2
            if num <= 6 and counts[suit*9 + num + 1] > 0 and counts[suit*9 + num + 2] > 0:
                tokens.append(CHI_OFFSET + chi_type_to_idx(suit, num + 1, 2))  # low=num+1 (1-based)

        # ── 荣和 ──────────────────────────────────────
        if not state.furiten[seat]:
            test_counts = list(counts)
            test_counts[pai.tile_id] += 1
            sht = calc_shanten(test_counts, len(state.melds[seat]))
            if sht == -1:
                tokens.append(RON_AGARI_TOKEN)

        if not tokens:
            return []  # 没有任何鸣牌机会，不生成决策点

        tokens.append(PASS_TOKEN)
        return tokens

    def _compute_discard_candidates(
        self, state: GameState, seat: int
    ) -> ActionCandidate:
        cans = ActionCandidate(can_discard=True)
        counts = hand_to_counts(state.hands[seat])
        meld_count = len(state.melds[seat])
        shanten = calc_shanten(counts, meld_count)

        # ── 立直中只能摸切或和了，不能打其他牌 ─────────────────────
        if state.in_riichi[seat]:
            # 候选动作只有：摸切（不入 candidates，在 _build_candidate_tokens 里单独处理）
            # 和/自摸 (shanten==-1)
            cans.discard_candidates = []  # 立直中不开放常规弃牌候选
            cans.can_riichi = False       # 已立直，不再重复宣言
            if shanten == -1:
                cans.can_tsumo = True
            return cans

        # 收集所有的合法打牌 tile_id（包括赤宝牌映射到的 34, 35, 36）
        cands_set = set()
        for tile in state.hands[seat]:
            if tile.is_aka:
                aka_idx = {4: 34, 13: 35, 22: 36}.get(tile.tile_id)
                if aka_idx is not None:
                    cands_set.add(aka_idx)
                else:
                    cands_set.add(tile.tile_id)
            else:
                cands_set.add(tile.tile_id)

        cans.discard_candidates = list(cands_set)
        # can_riichi: only when exactly tenpai (shanten==0) and not already in riichi
        if shanten == 0:
            cans.can_riichi = not state.in_riichi[seat]
        if shanten == -1:
            cans.can_tsumo = True
        return cans

    def _build_candidate_tokens(
        self, cans: ActionCandidate, primary: ActionType
    ) -> list[int]:
        from rinshan.constants import (
            DISCARD_OFFSET, RIICHI_TOKEN, TSUMO_AGARI_TOKEN,
            RON_AGARI_TOKEN, PASS_TOKEN, MAX_CANDIDATES_LEN,
        )
        tokens = []
        if cans.can_discard:
            for tid in sorted(cans.discard_candidates):
                tokens.append(DISCARD_OFFSET + tid)
        if cans.can_riichi:   tokens.append(RIICHI_TOKEN)
        if cans.can_tsumo:    tokens.append(TSUMO_AGARI_TOKEN)
        if cans.can_ron:      tokens.append(RON_AGARI_TOKEN)
        if cans.can_pass:     tokens.append(PASS_TOKEN)
        # 立直中 discard_candidates 为空，只有 tsumo（和了）或 PASS（摸切）
        if not tokens:
            tokens.append(PASS_TOKEN)
        return tokens[:MAX_CANDIDATES_LEN]

    def _find_action_idx(self, cands: list[int], action: Action) -> int:
        token = encode_action(action)
        for i, c in enumerate(cands):
            if c == token:
                return i
        # 找不到对应 token：记录警告并返回 0（第一个候选）
        import logging
        logging.getLogger(__name__).debug(
            f"_find_action_idx: token {token} not in cands {cands}, fallback to 0"
        )
        return 0

    def _make_annotation(
        self,
        game_id: str,
        seat: int,
        view: PlayerView,
        state: GameState,
        cands: list[int],
        chosen: int,
    ) -> Annotation:
        counts = hand_to_counts(view.hand)
        meld_count = len(view.melds[0]) if view.melds else 0
        sht = calc_shanten(counts, meld_count)

        aux = AuxTargets(
            shanten      = sht,
            tenpai_prob  = float(sht <= 0),
            deal_in_risk = self._calc_deal_in_risk(state, seat),
            opp_tenpai   = [
                int(state.in_riichi[(seat + i + 1) % 4]) for i in range(3)
            ],
        )

        return Annotation(
            game_id           = game_id,
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
            action_candidates = list(cands),
            action_chosen     = chosen,
            aux               = aux,
            # 三家对手手牌（全知视角，天凤日志里四家手牌均可见）
            # 旋转使 [0]=座位(seat+1)%4, [1]=(seat+2)%4, [2]=(seat+3)%4
            opponent_hands    = [
                list(state.hands[(seat + i + 1) % 4]) for i in range(3)
            ],
        )

    def _calc_deal_in_risk(self, state: GameState, seat: int) -> list[float]:
        """
        计算自家打出每张牌被荣和的风险度（Label B：对手待张集合）。

        risk[tile_id] = 听牌且未振听的对手中，tile 是其待张的对手数 / 3
        值域 = {0.0, 0.33, 0.67, 1.0}

        复盘牌谱四家手牌全程可见，精确计算作为训练标签；
        推理时模型仅凭公开信息预测此值，"?" guard 保留兼容性。
        """
        from rinshan.tile import hand_to_counts
        from rinshan.algo.shanten import calc_shanten

        # 对手人数（固定 3 家）
        N_OPP = 3
        # 累积每张牌的危险对手数
        danger_count = [0] * 34

        for opp in range(4):
            if opp == seat:
                continue

            opp_hand = state.hands[opp]
            if not opp_hand:
                continue

            # 对手已振听，无法荣和任何牌
            if state.furiten[opp]:
                continue

            # 对手已立直：手牌固定，直接枚举待张，跳过 n_tiles 分支判断
            if state.in_riichi[opp]:
                counts = hand_to_counts(opp_hand)
                mc = len(state.melds[opp])
                for tile_id in range(34):
                    counts[tile_id] += 1
                    if calc_shanten(counts, mc) == -1:
                        danger_count[tile_id] += 1
                    counts[tile_id] -= 1
                continue

            meld_count = len(state.melds[opp])
            counts = hand_to_counts(opp_hand)

            # 对手当前必须处于听牌状态（shanten == 0）
            # 注意：shanten 是基于「打出一张后」的 13 张手牌计算的
            # 自对弈中摸牌后手牌是 14 张，shanten=0 表示打出某张后可听牌
            # 但此处状态是决策前的完整手牌，已经是摸牌后的状态
            # 用 shanten(手牌当前张数) 判断：
            #   13张 shanten==0 → 已经听牌（等待荣和）
            #   14张 shanten==0 → 打出一张后听牌（还没选择打哪张）
            # 我们只关心「已经听牌」的对手，即 13 张时 shanten==0
            # 对手手牌张数 = 13（未摸牌）才真正「在等待荣和」
            n_tiles = sum(counts)
            if n_tiles == 14:
                # 对手刚摸牌（轮到对手打牌），还没确定打哪张
                # 保守处理：假设对手会打出最优牌，枚举打出每张后是否听牌
                # 简化：只要有一种打法能听牌，就认为对手「可能」在听牌
                # 这是高估，但 Stage 4 早期用 Oracle 精确值，宁高估不低估
                opponent_is_tenpai = False
                for t_id in range(34):
                    if counts[t_id] == 0:
                        continue
                    counts[t_id] -= 1
                    if calc_shanten(counts, meld_count) == 0:
                        opponent_is_tenpai = True
                        counts[t_id] += 1
                        break
                    counts[t_id] += 1
                if not opponent_is_tenpai:
                    continue
                # 重新计算 counts 用于待张枚举（用原始 14 张）
                counts = hand_to_counts(opp_hand)
                # 枚举对手打出每张后的待张（取所有打法的待张并集）
                waits = set()
                for t_id in range(34):
                    if counts[t_id] == 0:
                        continue
                    counts[t_id] -= 1
                    if calc_shanten(counts, meld_count) == 0:
                        # 这种打法下对手处于听牌，枚举其待张
                        for w in range(34):
                            counts[w] += 1
                            if calc_shanten(counts, meld_count) == -1:
                                waits.add(w)
                            counts[w] -= 1
                    counts[t_id] += 1
                for w in waits:
                    danger_count[w] += 1

            elif n_tiles == 13:
                # 对手已处于听牌等待状态（shanten==0 表示等待某张和了）
                if calc_shanten(counts, meld_count) != 0:
                    continue
                # 枚举 34 种牌，哪些是对手的待张
                for tile_id in range(34):
                    counts[tile_id] += 1
                    if calc_shanten(counts, meld_count) == -1:
                        danger_count[tile_id] += 1
                    counts[tile_id] -= 1

            # 其他张数（副露后手牌减少）同样处理
            else:
                if calc_shanten(counts, meld_count) != 0:
                    continue
                for tile_id in range(34):
                    counts[tile_id] += 1
                    if calc_shanten(counts, meld_count) == -1:
                        danger_count[tile_id] += 1
                    counts[tile_id] -= 1

        # 归一化：危险对手数 / 3 → [0.0, 0.33, 0.67, 1.0]
        return [danger_count[t] / N_OPP for t in range(34)]


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _remove_tile(hand: list[Tile], tile: Tile):
    """从手牌中移除一张牌（优先移除匹配赤标记的）"""
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id and t.is_aka == tile.is_aka:
            hand.pop(i)
            return
    # fallback：忽略赤标记
    for i, t in enumerate(hand):
        if t.tile_id == tile.tile_id:
            hand.pop(i)
            return


def _scores_to_ranks(scores: list[int]) -> list[int]:
    """分数 → 排名（0=一位，3=四位）"""
    order = sorted(range(4), key=lambda i: scores[i], reverse=True)
    ranks = [0] * 4
    for rank, seat in enumerate(order):
        ranks[seat] = rank
    return ranks
