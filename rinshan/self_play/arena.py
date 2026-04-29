"""
Arena — 并行自对弈调度器

参考 mortal/libriichi/src/arena/game.rs 的 BatchGame 模式：
  - 同时跑多局游戏，均摊 GPU 推理开销
  - 所有 AI agent 的推理可以批量化（对 RinshanAgent 尤为重要）
  - 产出 GameRecord，可直接转化为 mjai 事件流保存或用于在线训练

使用示例：
    arena = Arena(
        agents=[RinshanAgent(model), RandomAgent()],
        n_games=64,
        game_length="hanchan",
        seed=42,
    )
    records = arena.run()
    for rec in records:
        print(rec.final_scores, rec.ranks)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rinshan.engine.game_board import KyokuBoard, KyokuResult


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class GameRecord:
    """一场对局的完整记录"""
    game_id:      str
    seed:         tuple[int, int]
    agent_names:  list[str]           # 四家 agent 名称（按座位）
    final_scores: list[int]           # 终局分数
    ranks:        list[int]           # 0=一位 … 3=四位
    kyoku_logs:   list[list[dict]]    # 每局的 mjai 事件流

    @property
    def score_deltas(self) -> list[int]:
        return [s - 25000 for s in self.final_scores]


# ─────────────────────────────────────────────
# 单局包装
# ─────────────────────────────────────────────

class _GameRunner:
    """
    驱动一场完整对局（东风/半庄）。

    每次 step() 调用执行以下循环：
      1. 向 KyokuBoard 询问 pending decisions
      2. 通知对应 agent 提交响应
      3. 调用 board.resolve() 推进
      4. 若该局结束，处理结算并进入下一局
    """

    def __init__(
        self,
        game_id:    str,
        seed:       tuple[int, int],
        agents:     list,               # list[BaseAgent]，4 个座位的 agent
        game_length:int = 8,            # 8=半庄 4=东风
        init_scores:list[int] = None,
    ):
        self.game_id     = game_id
        self.seed        = seed
        self.agents      = agents       # 已按座位排好
        self.game_length = game_length
        self.scores      = list(init_scores or [25000]*4)

        self.kyoku   = 0
        self.honba   = 0
        self.kyotaku = 0
        self.in_renchan = False

        self.kyoku_logs: list[list[dict]] = []
        self._board: Optional[KyokuBoard] = None
        self.done = False
        self.result: Optional[GameRecord] = None

        for agent in self.agents:
            agent.on_game_start()

        self._start_kyoku()

    # ──────────────────────────────────────────

    def _game_over_condition(self) -> bool:
        """是否应结束整场"""
        # 超出局数上限（含延长局最多 +4）
        if self.kyoku >= self.game_length + 4:
            return True
        # 正常终局：超过基础局数，且有人超过 30000，且不在连庄
        if (self.kyoku >= self.game_length
                and not self.in_renchan
                and any(s >= 30000 for s in self.scores)):
            return True
        return False

    def _start_kyoku(self) -> None:
        if self._game_over_condition():
            self._finalize()
            return
        self._board = KyokuBoard(
            kyoku   = self.kyoku,
            honba   = self.honba,
            kyotaku = self.kyotaku,
            scores  = self.scores,
            game_seed = self.seed,
        )

    def _finalize(self) -> None:
        """游戏结束，生成 GameRecord"""
        self.done = True
        # 供托棒归入最高分玩家（天凤规则）
        if self.kyotaku > 0:
            top_seat = max(range(4), key=lambda i: self.scores[i])
            self.scores[top_seat] += self.kyotaku * 1000
            self.kyotaku = 0

        sorted_seats = sorted(range(4), key=lambda i: self.scores[i], reverse=True)
        ranks = [0] * 4
        for rank, seat in enumerate(sorted_seats):
            ranks[seat] = rank

        self.result = GameRecord(
            game_id      = self.game_id,
            seed         = self.seed,
            agent_names  = [a.name for a in self.agents],
            final_scores = list(self.scores),
            ranks        = ranks,
            kyoku_logs   = self.kyoku_logs,
        )
        for agent in self.agents:
            agent.on_game_end(self.result)

    def _handle_kyoku_end(self, board: KyokuBoard) -> None:
        """一局结束后的结算与状态更新"""
        res: KyokuResult = board.result
        self.kyoku_logs.append(board.get_log())

        for agent in self.agents:
            agent.on_kyoku_end()

        # 应用分数变化
        for i in range(4):
            self.scores[i] += res.deltas[i]

        # 飞出判断
        if any(s < 0 for s in self.scores):
            self._finalize()
            return

        # 更新供托
        if res.has_hora:
            self.kyotaku = 0
        else:
            self.kyotaku = res.kyotaku_left

        self.in_renchan = False

        if res.has_abortive_ryukyoku:
            self.honba += 1
            self._start_kyoku()
            return

        if not res.can_renchan:
            self.kyoku += 1
            self.honba = 0 if res.has_hora else self.honba + 1
            self._start_kyoku()
            return

        # 连庄终局条件
        oya = res.kyoku % 4
        if res.kyoku >= self.game_length - 1 and self.scores[oya] >= 30000:
            top_seat = max(range(4), key=lambda i: self.scores[i])
            if top_seat == oya:
                self._finalize()
                return

        # 连庄
        self.in_renchan = True
        self.honba += 1
        self._start_kyoku()

    def step(self) -> bool:
        """
        执行一个推进步骤。

        Returns:
            True  = 游戏已结束（self.done == True）
            False = 游戏仍在进行
        """
        if self.done:
            return True

        board = self._board
        if board is None or board.done:
            if board and board.done:
                self._handle_kyoku_end(board)
            return self.done

        if not board.has_pending():
            # 没有 pending，理论上不会出现（board 初始化后立刻有 pending）
            return False

        # 收集所有需要决策的 agent 响应
        batch_groups: dict[int, dict] = {}
        for seat in range(4):
            pending = board.pending_decisions[seat]
            if pending is None:
                continue
            player_log = board.get_player_log(seat)
            agent = self.agents[seat]
            pending = dict(pending)
            pending["_game_key"] = self.game_id  # 只用 game_id，避免每局换 key 导致缓存 miss

            if hasattr(agent, "react_batch_requests"):
                model_key = id(getattr(agent, "model", agent))
                if model_key not in batch_groups:
                    batch_groups[model_key] = {"agent": agent, "items": []}
                batch_groups[model_key]["items"].append((seat, player_log, pending))
            else:
                response = agent.react(seat, player_log, pending)
                board.push_reaction(seat, response)

        # 对支持批量推理的 agent 做批处理
        for group in batch_groups.values():
            agent = group["agent"]
            items = group["items"]
            responses = agent.react_batch_requests(items)
            for (seat, _player_log, _pending), response in zip(items, responses):
                board.push_reaction(seat, response)

        if board.ready_to_resolve():
            board.resolve()

        return self.done


# ─────────────────────────────────────────────
# Arena：并行调度多局游戏
# ─────────────────────────────────────────────

class Arena:
    """
    并行运行多场对局并收集 GameRecord。

    Args:
        agents:        参与的 agent 列表（可以是 1 个或多个，自动分配到四席）
        n_games:       要跑的总局数
        game_length:   8=半庄，4=东风
        seats_per_game:4（固定四人麻将）
        base_seed:     随机种子基础值（nonce=base_seed, key=game_idx）
        agent_rotation:如何把 agent 分配到四个座位
                         - "same"    : 四席用同一个 agent（单 agent 测试）
                         - "round_robin": 循环分配（多 agent 均衡对局）
        show_progress: 是否打印进度
    """

    def __init__(
        self,
        agents:         list,
        n_games:        int = 1,
        game_length:    str = "hanchan",
        base_seed:      int = 0,
        agent_rotation: str = "round_robin",
        show_progress:  bool = True,
    ):
        self.agents        = agents
        self.n_games       = n_games
        self.game_length   = 8 if game_length in ("hanchan", "half") else 4
        self.base_seed     = base_seed
        self.agent_rotation = agent_rotation
        self.show_progress  = show_progress

    def _assign_seats(self, game_idx: int) -> list:
        """根据 agent_rotation 策略，为 game_idx 分配四个座位的 agent"""
        n = len(self.agents)
        if self.agent_rotation == "same" or n == 1:
            return [self.agents[0]] * 4
        # round_robin：每局按游戏序号轮换座位
        seats = []
        for seat in range(4):
            agent_idx = (game_idx * 4 + seat) % n
            seats.append(self.agents[agent_idx])
        return seats

    def run(self) -> list[GameRecord]:
        """
        顺序执行所有对局（多局并行的调度在此函数内）。

        返回 list[GameRecord]，长度 == n_games。
        """
        records: list[GameRecord] = []
        t0 = time.time()

        runners = []
        for game_idx in range(self.n_games):
            seed = (self.base_seed + game_idx, 0)
            game_id = f"sp_{self.base_seed}_{game_idx:06d}"
            seat_agents = self._assign_seats(game_idx)
            runners.append(_GameRunner(
                game_id     = game_id,
                seed        = seed,
                agents      = seat_agents,
                game_length = self.game_length,
            ))

        # 主调度循环：跨 runner 收集 pending，请求支持 batch 的 agent 统一推理。
        active = list(runners)
        while active:
            still_active = []
            batch_groups: dict[int, dict] = {}

            for runner in active:
                if runner.done:
                    if runner.result:
                        records.append(runner.result)
                    continue

                board = runner._board
                if board is None or board.done:
                    if board and board.done:
                        runner._handle_kyoku_end(board)
                    if runner.done:
                        if runner.result:
                            records.append(runner.result)
                        continue
                    board = runner._board

                if board is None:
                    continue

                still_active.append(runner)

                if not board.has_pending():
                    continue

                for seat in range(4):
                    pending = board.pending_decisions[seat]
                    if pending is None:
                        continue
                    player_log = board.get_player_log(seat)
                    agent = runner.agents[seat]
                    pending = dict(pending)
                    # 只用 game_id 做 key：_get_cached_state 通过事件数增量判断是否需要重建，
                    # 加入 kyoku/honba 后每局换局时 key 必然 miss，强制全量重放，反而更慢。
                    pending["_game_key"] = runner.game_id

                    if hasattr(agent, "react_batch_requests"):
                        # 按 id(agent.model) 分组：共享同一模型的 agent
                        # （如 versus 模式的 ch_0/ch_1）合并入同一 batch，
                        # 避免 batch size 被不必要地切半。
                        model_key = id(getattr(agent, "model", agent))
                        if model_key not in batch_groups:
                            batch_groups[model_key] = {"agent": agent, "items": [], "targets": []}
                        batch_groups[model_key]["items"].append((seat, player_log, pending))
                        batch_groups[model_key]["targets"].append((runner, seat))
                    else:
                        response = agent.react(seat, player_log, pending)
                        board.push_reaction(seat, response)

            # 对支持批量推理的 agent 跨局统一推理
            for group in batch_groups.values():
                agent = group["agent"]
                items = group["items"]
                targets = group["targets"]
                responses = agent.react_batch_requests(items)
                for (runner, seat), response in zip(targets, responses):
                    if runner._board is not None:
                        runner._board.push_reaction(seat, response)

            # 推进所有已收齐响应的对局
            for runner in still_active:
                board = runner._board
                if board is not None and board.ready_to_resolve():
                    board.resolve()

            if self.show_progress:
                elapsed = time.time() - t0
                n_done = len(records)
                speed = n_done / elapsed if elapsed > 0 else 0
                print(f"\r[Arena] {n_done}/{self.n_games} games ({speed:.2f} games/s)", end="", flush=True)

            active = still_active

        if self.show_progress:
            print()  # 换行
        return records

    def run_parallel(self, n_workers: int = 4) -> list[GameRecord]:
        """
        多进程并行版本（适合 CPU-only 或多 GPU 场景）。
        对于 GPU agent（RinshanAgent），推荐用批量化推理而非多进程，
        此方法更适合 RandomAgent 等纯 CPU 的场景。
        """
        import multiprocessing as mp
        from functools import partial

        def _run_single(game_idx: int, base_seed: int,
                        game_length: int) -> GameRecord:
            seed = (base_seed + game_idx, 0)
            game_id = f"sp_{base_seed}_{game_idx:06d}"
            # 注意：多进程中 agent 需要可序列化，RandomAgent 满足此要求
            seat_agents = self._assign_seats(game_idx)
            runner = _GameRunner(
                game_id     = game_id,
                seed        = seed,
                agents      = seat_agents,
                game_length = game_length,
            )
            while not runner.done:
                runner.step()
            return runner.result

        fn = partial(_run_single,
                     base_seed=self.base_seed,
                     game_length=self.game_length)

        with mp.Pool(n_workers) as pool:
            records = pool.map(fn, range(self.n_games))

        return [r for r in records if r is not None]
