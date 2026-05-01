"""
mjlog_parser.py — 天凤 mjlog XML → mjai JSON 事件流

mjlog 格式说明：
  - 牌编号: tile_id 0-135，tile_id//4 得牌型(0-33)，16=赤5m, 52=赤5p, 88=赤5s
  - 摸/打牌: T=玩家0摸, D=玩家0打, U=玩家1摸, E=玩家1打, V=玩家2摸, F=玩家2打, W=玩家3摸, G=玩家3打
  - 鸣牌: <N who="x" m="..."/> m 字段是 16-bit 编码
  - INIT: 局开始，含手牌/分数/宝牌/庄家
  - REACH: 立直 (step=1 宣言, step=2 扣点确认)
  - AGARI: 和牌
  - RYUUKYOKU: 流局

输出 mjai 格式事件列表，可直接喂给 MjaiSimulator。
"""
from __future__ import annotations

import re
import urllib.error
import urllib.parse
from typing import Optional
import xml.etree.ElementTree as ET


# ─────────────────────────────────────────────
# 牌编号转换
# ─────────────────────────────────────────────

# mjlog tile_id → mjai 字符串
_TYPE_TO_MJAI = (
    # man
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    # pin
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    # sou
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    # honors
    "1z","2z","3z","4z","5z","6z","7z",
)

# 赤宝牌 tile_id
_AKA_IDS = {16, 52, 88}   # 赤5m, 赤5p, 赤5s
_AKA_MJAI = {16: "0m", 52: "0p", 88: "0s"}


def mjlog_tile_to_mjai(tile_id: int) -> str:
    """将 mjlog tile_id 转为 mjai 字符串"""
    if tile_id in _AKA_MJAI:
        return _AKA_MJAI[tile_id]
    return _TYPE_TO_MJAI[tile_id // 4]


def mjlog_tiles_to_mjai(tile_ids: list[int]) -> list[str]:
    return [mjlog_tile_to_mjai(t) for t in tile_ids]


# ─────────────────────────────────────────────
# 鸣牌 m 值解码
# ─────────────────────────────────────────────

def decode_meld(m: int) -> dict:
    """
    解码 mjlog 的鸣牌 m 字段（16-bit），返回 mjai 格式的鸣牌信息。

    参考：天凤 mjlog 格式文档 + mjlogevent.py 实现
    返回：{"type": "chi"|"pon"|"daiminkan"|"ankan"|"kakan",
            "pai": "...",          # 被鸣的牌（chi/pon/kan）
            "consumed": [...],     # 从手牌消耗的牌
           }
    """
    if m & 0x4:    # 吃 (chi)：bit2 优先
        return _decode_chi(m)
    if m & 0x18:   # 碰 (bit3) or 加杠 (bit4)
        return _decode_pon_or_kakan(m)
    if m & 0x20:   # 拔北 (北出し)
        t = (m >> 8) >> 2
        tile = _TYPE_TO_MJAI[t]
        return {"type": "nukidora", "pai": tile, "consumed": [tile]}
    # 其他：暗杠 or 大明杠
    return _decode_kan(m)


def _decode_chi(m: int) -> dict:
    """吃牌解码"""
    # t0 = 叫牌索引的基准
    t0 = (m >> 10) // 3
    t  = t0 + (t0 // 7) * 2  # 调整跨越字牌的偏移（chi 不含字牌，此处防护）

    # 三张牌的类型索引（相对于顺子起始）
    r = (m >> 10) % 3     # 被鸣的牌在顺子中的位置（0=低,1=中,2=高）

    # 恢复三张牌的 tile_id
    tiles = []
    for i in range(3):
        base_type = t + i
        kui_bit = (m >> (3 + i * 2)) & 0x3
        tid = base_type * 4 + kui_bit
        tiles.append(tid)

    called_tile = tiles[r]
    consumed = [tiles[i] for i in range(3) if i != r]

    return {
        "type":     "chi",
        "pai":      mjlog_tile_to_mjai(called_tile),
        "consumed": mjlog_tiles_to_mjai(consumed),
    }


def _decode_pon_or_kakan(m: int) -> dict:
    """碰 / 加杠解码"""
    # 区分碰和加杠：bit 4 (0x10) 为加杠标记
    is_kakan = bool(m & 0x10)
    t  = (m >> 9) // 3
    r  = (m >> 9) % 3  # 被碰走的那张在 t*4+0~3 中的位置
    t4 = t * 4

    # 实际被鸣的牌（用 r 找被打出的那张）
    called_tile_id = t4 + r
    # 手里的两张
    remaining = [t4 + i for i in range(4) if i != r]

    if is_kakan:
        # 加杠：手里再多出一张
        consumed = mjlog_tiles_to_mjai(remaining[:2])
        return {
            "type":     "kakan",
            "pai":      mjlog_tile_to_mjai(called_tile_id),
            "consumed": consumed,
        }
    else:
        consumed = mjlog_tiles_to_mjai(remaining[:2])
        return {
            "type":     "pon",
            "pai":      mjlog_tile_to_mjai(called_tile_id),
            "consumed": consumed,
        }


def _decode_kan(m: int) -> dict:
    """
    大明杠 / 暗杠解码

    天凤 mjlog 杠牌编码规则：
      t = (m & 0xFF00) >> 8   (高 8 位)
      tile_type = t >> 2      (0-33)
      t & 3 == 0  → 暗杠 (ankan)，4 张全从手牌
      t & 3 != 0  → 大明杠 (daiminkan)，被打出牌的位置在 t & 3
    """
    t = (m & 0xFF00) >> 8       # 高8位
    tile_type = t >> 2           # 牌型索引 0-33
    base_tid  = tile_type * 4    # 该牌型的基础 tile_id

    if t & 3 == 0:  # 暗杠
        pai_mjai = mjlog_tile_to_mjai(base_tid)
        return {
            "type":     "ankan",
            "pai":      pai_mjai,
            "consumed": [pai_mjai] * 4,
        }
    else:           # 大明杠
        called_tid = base_tid + (t & 3)
        pai_mjai   = mjlog_tile_to_mjai(called_tid)
        consumed   = [mjlog_tile_to_mjai(base_tid + i) for i in range(4) if (base_tid + i) != called_tid]
        return {
            "type":     "daiminkan",
            "pai":      pai_mjai,
            "consumed": consumed,
        }


# ─────────────────────────────────────────────
# 和牌解析辅助
# ─────────────────────────────────────────────

def _parse_score_changes(sc_str: str) -> list[int]:
    """
    解析 AGARI/RYUUKYOKU 的 sc 字段
    格式: "before0,delta0,before1,delta1,..."
    返回: [delta0, delta1, delta2, delta3]
    """
    vals = [int(x) for x in sc_str.split(",")]
    # vals[0],vals[1] = p0_before, p0_delta
    return [vals[i*2+1] for i in range(4)]


# ─────────────────────────────────────────────
# 主解析器
# ─────────────────────────────────────────────

# 摸牌/打牌 tag 映射
_DRAW_TAGS    = {"T": 0, "U": 1, "V": 2, "W": 3}
_DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}


class MjlogParser:
    """
    解析单局 mjlog XML，输出 mjai 格式事件列表。

    Usage:
        parser = MjlogParser()
        events = parser.parse(xml_text)
        # events 是 list[dict]，可直接传给 MjaiSimulator.parse_game()
    """

    def parse(self, xml_text: str, game_id: str = "unknown") -> list[dict]:
        """
        解析完整的 mjlog XML（一个 <mjloggm> 根元素）
        返回 mjai 格式事件列表
        """
        # 天凤 XML 不规范，手动拆 tag 比 ET.parse 更可靠
        events: list[dict] = [{"type": "start_game", "id": game_id}]

        # 匹配所有 XML 标签（自闭合 + 开闭标签）
        tag_pat = re.compile(r'<([A-Z][A-Z0-9_]*)([^/]*?)/?>', re.DOTALL)
        # 也匹配单字母 draw/discard tags: <T64/> <D134/>
        dt_pat  = re.compile(r'<([TDUVEFW]G?)(\d*)/?>')

        # 合并两种 tag 的扫描
        all_tags = list(re.finditer(
            r'<([A-Za-z][A-Za-z0-9_]*)([^/]*)/?>', xml_text
        ))

        game_info: dict = {}  # UN 信息：玩家名等
        round_idx = 0
        honba_now = 0
        kyotaku_now = 0
        current_scores = [250, 250, 250, 250]  # 以 250 代表 25000 点（/100）
        # last_draw_tile_id[seat] = 该玩家最近摸牌的 raw mjlog tile_id（未经 deaka）
        # tsumogiri 判断：打出的 tile_id == last_draw_tile_id[seat] → 摸切
        # 覆盖所有情况：普通摸切/手切、摸切立直/手切立直、立直后续摸切、鸣牌后出牌
        last_draw_tile_id: dict = {0: None, 1: None, 2: None, 3: None}

        for m in all_tags:
            tag = m.group(1).upper()
            attrs_str = m.group(2)
            attrs = _parse_attrs(attrs_str)

            # ── GO：游戏类型（忽略详情，仅记录）──
            if tag == "GO":
                game_info["type"] = attrs.get("type", "")
                continue

            # ── UN：玩家信息 ──────────────────────
            if tag == "UN":
                names = []
                for i in range(4):
                    n = attrs.get(f"n{i}", "")
                    names.append(urllib.parse.unquote(n))
                game_info["names"] = names
                continue

            # ── TAIKYOKU：游戏开始 ───────────────
            if tag == "TAIKYOKU":
                continue

            # ── INIT：局开始 ─────────────────────
            if tag == "INIT":
                round_idx += 1
                seed = attrs.get("seed", "0,0,0,0,0,0").split(",")
                honba_now   = int(seed[1]) if len(seed) > 1 else 0
                kyotaku_now = int(seed[2]) if len(seed) > 2 else 0
                # dora indicator tile_id
                dora_tile_id = int(seed[5]) if len(seed) > 5 else 0

                ten = [int(x) for x in attrs.get("ten", "250,250,250,250").split(",")]
                current_scores = [t * 100 for t in ten]  # 转为实际点数

                oya = int(attrs.get("oya", "0"))

                # 手牌：天凤牌谱是完整复盘，hai0~hai3 四家都有明文
                tehais = []
                for i in range(4):
                    hai_str = attrs.get(f"hai{i}", "")
                    if hai_str:
                        tiles = [mjlog_tile_to_mjai(int(t)) for t in hai_str.split(",")]
                    else:
                        tiles = []  # 数据缺失时给空列表，simulator 会跳过
                    tehais.append(tiles)

                # 场风/局数
                round_wind, round_num = divmod(oya + round_idx - 1, 4)
                # 更简单直接：从 seed[0] 读局数
                kyoku_raw = int(seed[0]) if seed[0].isdigit() else 0
                # seed[0] = 绝对局数（E1=0, E2=1, ..., S1=4 ...）
                # 但不同版本含义可能不同，用 oya 位移更可靠
                # 此处保持简单：用累计 round_idx 估算
                bakaze_map = {0: "E", 1: "S", 2: "W", 3: "N"}
                # 用 seed[0] 解析更准确
                abs_kyoku = int(seed[0]) if len(seed) > 0 and seed[0].strip().isdigit() else 0
                baze  = bakaze_map.get(abs_kyoku // 4, "E")
                kyoku = abs_kyoku % 4 + 1

                events.append({
                    "type":         "start_kyoku",
                    "bakaze":       baze,
                    "kyoku":        kyoku,
                    "honba":        honba_now,
                    "kyotaku":      kyotaku_now,
                    "oya":          oya,
                    "dora_marker":  mjlog_tile_to_mjai(dora_tile_id),
                    "scores":       list(current_scores),
                    "tehais":       tehais,
                })
                last_draw_tile_id = {0: None, 1: None, 2: None, 3: None}
                continue

            # ── 摸牌：T/U/V/W ────────────────────
            draw_match = re.match(r'^([TUVW])(\d+)$', tag + attrs_str.strip())
            if tag[0] in "TUVW" and len(tag) == 1:
                try:
                    tile_id = int(attrs_str.strip().strip('/').strip() or "0")
                    # 实际 tag 格式是 <T64/> 这样，tag="T", attrs_str="64"
                    # 但 re 匹配可能把数字放在 attrs_str 里
                    if not attrs_str.strip():
                        # 数字在 tag 名里，如 <T64/>
                        num_match = re.match(r'([TUVW])(\d+)', m.group(0))
                        if num_match:
                            tag2 = num_match.group(1)
                            tile_id = int(num_match.group(2))
                            seat = {"T":0,"U":1,"V":2,"W":3}[tag2]
                        else:
                            continue
                    else:
                        seat = {"T":0,"U":1,"V":2,"W":3}.get(tag[0], -1)
                    if seat >= 0:
                        last_draw_tile_id[seat] = tile_id
                    events.append({
                        "type":  "tsumo",
                        "actor": seat,
                        "pai":   mjlog_tile_to_mjai(tile_id) if seat >= 0 else "?",
                    })
                except (ValueError, KeyError):
                    pass
                continue

            # ── 打牌：D/E/F/G ────────────────────
            if tag[0] in "DEFG" and len(tag) == 1:
                try:
                    if not attrs_str.strip():
                        num_match = re.match(r'([DEFG])(\d+)', m.group(0))
                        if num_match:
                            tag2 = num_match.group(1)
                            tile_id = int(num_match.group(2))
                            seat = {"D":0,"E":1,"F":2,"G":3}[tag2]
                        else:
                            continue
                    else:
                        seat = {"D":0,"E":1,"F":2,"G":3}.get(tag[0], -1)
                        tile_id = int(attrs_str.strip())
                    # 通用判断：打出的 tile_id == 该玩家上一次摸牌的 tile_id → 摸切
                    # 覆盖所有情况：普通摸切、手切立直、摸切立直、立直后续摸切
                    tsumogiri = (seat >= 0 and last_draw_tile_id.get(seat) == tile_id)
                    if seat >= 0:
                        last_draw_tile_id[seat] = None  # 打牌后清空
                    events.append({
                        "type":      "dahai",
                        "actor":     seat,
                        "pai":       mjlog_tile_to_mjai(tile_id) if seat >= 0 else "?",
                        "tsumogiri": tsumogiri,
                    })
                except (ValueError, KeyError):
                    pass
                continue

            # ── 鸣牌：N ──────────────────────────
            if tag == "N":
                who = int(attrs.get("who", "0"))
                m_val = int(attrs.get("m", "0"))
                meld = decode_meld(m_val)
                meld["actor"] = who
                # target: 被鸣的玩家（chi=上家, pon/kan=任意）
                meld["target"] = (who - 1) % 4  # 简化：默认上家
                # 鸣牌后出牌不算摸切（没有摸牌动作）
                last_draw_tile_id[who] = None
                events.append({"type": meld["type"], **meld})
                continue

            # ── 立直：REACH ───────────────────────
            if tag == "REACH":
                who  = int(attrs.get("who", "0"))
                step = int(attrs.get("step", "1"))
                if step == 1:
                    events.append({"type": "reach", "actor": who})
                else:
                    # step=2: 扣 1000 点确认
                    ten_str = attrs.get("ten", "")
                    if ten_str:
                        new_ten = [int(x) * 100 for x in ten_str.split(",")]
                        current_scores = new_ten
                    events.append({
                        "type":   "reach_accepted",
                        "actor":  who,
                        "scores": list(current_scores),
                    })
                continue

            # ── 新宝牌：DORA ──────────────────────
            if tag == "DORA":
                hai = int(attrs.get("hai", "0"))
                events.append({
                    "type":        "dora",
                    "dora_marker": mjlog_tile_to_mjai(hai),
                })
                continue

            # ── 和牌：AGARI ───────────────────────
            if tag == "AGARI":
                who      = int(attrs.get("who", "0"))
                from_who = int(attrs.get("fromWho", str(who)))
                sc_str   = attrs.get("sc", "")
                deltas   = _parse_score_changes(sc_str) if sc_str else [0, 0, 0, 0]

                # 更新分数
                for i in range(4):
                    current_scores[i] += deltas[i]

                # 检查本局终了（owari 字段存在则整场结束）
                owari = attrs.get("owari", "")

                events.append({
                    "type":      "hora",
                    "actor":     who,
                    "target":    from_who,
                    "pai":       mjlog_tile_to_mjai(int(attrs.get("machi", "0"))) if attrs.get("machi") else "?",
                    "deltas":    deltas,
                    "scores":    list(current_scores),
                    "is_tsumo":  who == from_who,
                })

                if owari:
                    # 整场结束
                    final = _parse_owari_scores(owari)
                    events.append({
                        "type":         "end_kyoku",
                        "final_scores": current_scores,
                        "final_pts":    final,
                    })
                continue

            # ── 流局：RYUUKYOKU ───────────────────
            if tag == "RYUUKYOKU":
                sc_str = attrs.get("sc", "")
                deltas = _parse_score_changes(sc_str) if sc_str else [0, 0, 0, 0]
                for i in range(4):
                    current_scores[i] += deltas[i]

                # 解析听牌手牌（流局听牌获得奖励）
                tenpai_hands = {}
                for i in range(4):
                    hai_str = attrs.get(f"hai{i}", "")
                    if hai_str:
                        tenpai_hands[i] = [mjlog_tile_to_mjai(int(t)) for t in hai_str.split(",")]

                events.append({
                    "type":         "ryuukyoku",
                    "deltas":       deltas,
                    "scores":       list(current_scores),
                    "tenpai_hands": tenpai_hands,
                })
                continue

        events.append({"type": "end_game"})
        return events


def _parse_attrs(attrs_str: str) -> dict[str, str]:
    """解析 XML 属性字符串 → dict"""
    result = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', attrs_str):
        result[m.group(1)] = m.group(2)
    return result


def _parse_owari_scores(owari: str) -> list[float]:
    """
    解析 AGARI 的 owari 字段：'142,-35.8,393,49.3,...'
    格式: score0,pt0,score1,pt1,...
    返回: [pt0, pt1, pt2, pt3]
    """
    vals = owari.split(",")
    return [float(vals[i * 2 + 1]) for i in range(4) if i * 2 + 1 < len(vals)]


# ─────────────────────────────────────────────
# 修复：正确处理 <T64/> 风格的 tag
# ─────────────────────────────────────────────

class MjlogParserV2(MjlogParser):
    """
    修正版解析器：正确处理天凤 mjlog 中
    摸牌/打牌 tag 的数字嵌在 tag 名里的情况。
    例：<T64/> 而不是 <T tile="64"/>
    """

    def parse(self, xml_text: str, game_id: str = "unknown") -> list[dict]:
        # 预处理：把 <T64/> → <DRAW who="0" tile="64"/>
        def normalize(match):
            full = match.group(0)
            letter = match.group(1)
            num    = match.group(2)
            actor_map = {"T":"0","U":"1","V":"2","W":"3",
                         "D":"0","E":"1","F":"2","G":"3"}
            act = actor_map.get(letter.upper(), "0")
            is_draw = letter.upper() in "TUVW"
            tag_name = "MJDRAW" if is_draw else "MJDISCARD"
            return f'<{tag_name} who="{act}" tile="{num}"/>'

        normalized = re.sub(r'<([TUVWDEFGtuvwdefg])(\d+)\s*/?>', normalize, xml_text)

        # 现在用父类逻辑，但需要重载 tag 处理
        return self._parse_normalized(normalized, game_id)

    def _parse_normalized(self, xml_text: str, game_id: str) -> list[dict]:
        events: list[dict] = [{"type": "start_game", "id": game_id}]

        all_tags = list(re.finditer(r'<([A-Z][A-Z0-9_]*)([^/]*)/?>', xml_text))

        current_scores = [25000, 25000, 25000, 25000]
        # last_draw_tile_id[seat]: 该玩家最近一次摸牌的 raw mjlog tile_id
        # 用于 tsumogiri 判断，鸣牌和新局时清空
        last_draw_tile_id: dict = {0: None, 1: None, 2: None, 3: None}

        for m in all_tags:
            tag      = m.group(1).upper()
            attrs    = _parse_attrs(m.group(2))

            if tag == "GO":
                continue

            if tag == "UN":
                continue

            if tag == "TAIKYOKU":
                continue

            if tag == "INIT":
                seed = attrs.get("seed", "0,0,0,0,0,0").split(",")
                honba_now   = int(seed[1]) if len(seed) > 1 else 0
                kyotaku_now = int(seed[2]) if len(seed) > 2 else 0
                dora_tile_id = int(seed[5]) if len(seed) > 5 else 0
                abs_kyoku    = int(seed[0]) if len(seed) > 0 and seed[0].strip().isdigit() else 0

                ten = [int(x) for x in attrs.get("ten", "250,250,250,250").split(",")]
                current_scores = [t * 100 for t in ten]
                oya  = int(attrs.get("oya", "0"))

                tehais = []
                for i in range(4):
                    hai_str = attrs.get(f"hai{i}", "")
                    if hai_str:
                        tiles = [mjlog_tile_to_mjai(int(t)) for t in hai_str.split(",")]
                    else:
                        tiles = []  # 数据缺失时给空列表，simulator 会跳过
                    tehais.append(tiles)

                baze_map = {0:"E", 1:"S", 2:"W", 3:"N"}
                baze  = baze_map.get(abs_kyoku // 4, "E")
                kyoku = abs_kyoku % 4 + 1

                events.append({
                    "type":        "start_kyoku",
                    "bakaze":      baze,
                    "kyoku":       kyoku,
                    "honba":       honba_now,
                    "kyotaku":     kyotaku_now,
                    "oya":         oya,
                    "dora_marker": mjlog_tile_to_mjai(dora_tile_id),
                    "scores":      list(current_scores),
                    "tehais":      tehais,
                })
                last_draw_tile_id = {0: None, 1: None, 2: None, 3: None}
                continue

            if tag == "MJDRAW":
                seat    = int(attrs.get("who", "0"))
                tile_id = int(attrs.get("tile", "0"))
                last_draw_tile_id[seat] = tile_id
                events.append({"type": "tsumo", "actor": seat,
                                "pai": mjlog_tile_to_mjai(tile_id)})
                continue

            if tag == "MJDISCARD":
                seat    = int(attrs.get("who", "0"))
                tile_id = int(attrs.get("tile", "0"))
                # 通用判断：打出 tile_id == 上次摸牌 tile_id → 摸切
                tsumogiri = (last_draw_tile_id.get(seat) == tile_id)
                last_draw_tile_id[seat] = None  # 打牌后清空
                events.append({"type": "dahai", "actor": seat,
                                "pai": mjlog_tile_to_mjai(tile_id),
                                "tsumogiri": tsumogiri})
                continue

            if tag == "N":
                who   = int(attrs.get("who", "0"))
                m_val = int(attrs.get("m", "0"))
                meld  = decode_meld(m_val)
                # 鸣牌后出牌不算摸切（没有摸牌动作）
                last_draw_tile_id[who] = None
                ev = {
                    "type":     meld["type"],
                    "actor":    who,
                    "target":   (who - 1) % 4,
                    "pai":      meld["pai"],
                    "consumed": meld.get("consumed", []),
                }
                events.append(ev)
                continue

            if tag == "REACH":
                who  = int(attrs.get("who", "0"))
                step = int(attrs.get("step", "1"))
                if step == 1:
                    events.append({"type": "reach", "actor": who})
                else:
                    ten_str = attrs.get("ten", "")
                    if ten_str:
                        current_scores = [int(x)*100 for x in ten_str.split(",")]
                    events.append({"type": "reach_accepted", "actor": who,
                                   "scores": list(current_scores)})
                continue

            if tag == "DORA":
                hai = int(attrs.get("hai", "0"))
                events.append({"type": "dora", "dora_marker": mjlog_tile_to_mjai(hai)})
                continue

            if tag == "AGARI":
                who      = int(attrs.get("who", "0"))
                from_who = int(attrs.get("fromWho", str(who)))
                sc_str   = attrs.get("sc", "")
                deltas   = _parse_score_changes(sc_str) if sc_str else [0,0,0,0]
                for i in range(4):
                    current_scores[i] += deltas[i]
                owari = attrs.get("owari", "")
                events.append({
                    "type":      "hora",
                    "actor":     who,
                    "target":    from_who,
                    "pai":       mjlog_tile_to_mjai(int(attrs["machi"])) if "machi" in attrs else "?",
                    "deltas":    deltas,
                    "scores":    list(current_scores),
                    "is_tsumo":  who == from_who,
                })
                if owari:
                    events.append({"type": "end_kyoku",
                                   "final_scores": current_scores,
                                   "final_pts": _parse_owari_scores(owari)})
                continue

            if tag == "RYUUKYOKU":
                sc_str = attrs.get("sc", "")
                deltas = _parse_score_changes(sc_str) if sc_str else [0,0,0,0]
                for i in range(4):
                    current_scores[i] += deltas[i]
                tenpai = {}
                for i in range(4):
                    hs = attrs.get(f"hai{i}", "")
                    if hs:
                        tenpai[i] = [mjlog_tile_to_mjai(int(t)) for t in hs.split(",")]
                events.append({"type": "ryuukyoku", "deltas": deltas,
                                "scores": list(current_scores), "tenpai_hands": tenpai})
                continue

        events.append({"type": "end_game"})
        return events
