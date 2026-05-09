import torch
from rinshan.training.losses import iql_loss
import inspect

sig = inspect.signature(iql_loss)
assert "riichi_action_mask" in sig.parameters, "param missing"

B, N = 4, 15
q       = torch.randn(B, N)
v       = torch.randn(B)
v_next  = torch.randn(B)
action  = torch.randint(0, N - 1, (B,))
reward  = torch.randn(B)
done    = torch.zeros(B, dtype=torch.bool)
q_tgt   = torch.randn(B)

# 1. no mask
loss1, d1 = iql_loss(q, v, v_next, action, reward, done, q_tgt, bc_weight=0.2)

# 2. mask first two samples as RIICHI
mask = torch.tensor([True, False, True, False])
loss2, d2 = iql_loss(q, v, v_next, action, reward, done, q_tgt,
                     bc_weight=0.2, riichi_action_mask=mask)
assert "riichi_bc_exempt" in d2, "exempt key missing"
assert int(d2["riichi_bc_exempt"]) == 2, f"expected 2, got {d2['riichi_bc_exempt']}"

# 3. all-zero mask must be identical to no mask
mask_z = torch.zeros(B, dtype=torch.bool)
loss3, _ = iql_loss(q, v, v_next, action, reward, done, q_tgt,
                    bc_weight=0.2, riichi_action_mask=mask_z)
assert abs(loss3.item() - loss1.item()) < 1e-5, "zero-mask must match no-mask"

print("iql_loss riichi_bc_exempt:", int(d2["riichi_bc_exempt"]), "  OK")
print("zero-mask identity check:  OK")
print("All assertions passed.")
