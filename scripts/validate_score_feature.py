import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))

from rinshan.constants import VOCAB_SIZE, SCORE_OFFSET, MAX_GAME_META_LEN, MAX_SEQ_LEN, \
    MAX_DORA_LEN, MAX_HAND_LEN, MAX_MELD_LEN, MAX_PROGRESSION_LEN, MAX_CANDIDATES_LEN
print(f"VOCAB_SIZE={VOCAB_SIZE}")
print(f"SCORE_OFFSET={SCORE_OFFSET}  covers {SCORE_OFFSET}~{SCORE_OFFSET+63}")
print(f"MAX_GAME_META_LEN={MAX_GAME_META_LEN}")
print(f"MAX_SEQ_LEN={MAX_SEQ_LEN}")
assert SCORE_OFFSET + 64 == VOCAB_SIZE, f"gap={VOCAB_SIZE - SCORE_OFFSET - 64}, SCORE_OFFSET={SCORE_OFFSET}, VOCAB_SIZE={VOCAB_SIZE}"
print("token range OK")

from rinshan.data.encoder import score_to_token, calc_riichi_shaping
print(f"score -60000 seat0 -> {score_to_token(-60000, 0)} (expect {SCORE_OFFSET+0})")
print(f"score  90000 seat0 -> {score_to_token(90000,  0)} (expect {SCORE_OFFSET+15})")
print(f"score  25000 seat1 -> {score_to_token(25000,  1)} (expect {SCORE_OFFSET+16+8})")

class MockAnn:
    dora_indicators = [1, 2]
s = calc_riichi_shaping(MockAnn())
print(f"riichi_shaping(2 dora)={s:.3f}  expect={(2*500+0.25*1500)/1000:.3f}")

from rinshan.data.encoder import GameEncoder
from rinshan.data.annotation import Annotation
from rinshan.tile import Tile

ann = Annotation(
    game_id="test", player_id=0, round_wind=0, round_num=1,
    honba=0, kyotaku=0,
    scores=[25000, 25000, 25000, 25000], tiles_left=60,
    hand=[Tile(i, False) for i in range(13)],
    dora_indicators=[Tile(0, False)],
    discards=[[], [], [], []], melds=[[], [], [], []],
    riichi_declared=[False]*4,
    progression=[],
    action_candidates=[37, 38, 39, 497],
    action_chosen=3,
)
enc = GameEncoder().encode(ann)
ACTUAL_SEQ_LEN = MAX_GAME_META_LEN + MAX_DORA_LEN + MAX_HAND_LEN + MAX_MELD_LEN + MAX_PROGRESSION_LEN + MAX_CANDIDATES_LEN
assert enc["tokens"].shape[0] == ACTUAL_SEQ_LEN, f"got {enc['tokens'].shape[0]}, expect {ACTUAL_SEQ_LEN}"
assert enc["is_riichi_action"] is True, enc["is_riichi_action"]
expected_score_tok = score_to_token(25000, 0)
assert enc["tokens"][6].item() == expected_score_tok, \
    f"score token mismatch: {enc['tokens'][6].item()} vs {expected_score_tok}"
print(f"tokens shape: {enc['tokens'].shape}  OK")
print(f"is_riichi_action: {enc['is_riichi_action']}  OK")
print(f"riichi_shaping: {enc['riichi_shaping']:.3f}  OK")
print("ALL OK")
