# Unitree Go2 — Controller abort (`controller`)

Read the Go2's handheld remote and latch a clean **software abort** on any button press. Module:
[`go2_remote.py`](go2_remote.py) → `Go2Remote`.

> ⚠️ This is a clean software abort, **not** an emergency stop. It stops the routine and holds stance.
> The firmware damping and the physical power/e-stop are the real stop and work even if this process is
> hung — keep them in reach. See [`SAFETY.md`](../../../SAFETY.md).

## The remote struct

The remote rides in `LowState_.wireless_remote` — a 40-byte `xRockerBtnDataStruct`:

| bytes | field |
|---|---|
| `[0:2]` | header |
| `[2:4]` | `uint16` button bitmask (little-endian) |
| `[4:20]` | three `float32` axes (`lx`, `rx`, `ry`) |
| `[20:24]` | one `float32` axis (`ly`) |

Button bit order (bit i = index i): `R1 L1 start select R2 L2 F1 F2 A B X Y up right down left`.

This is the **same struct the G1 uses**, so `parse_buttons` / `parse_sticks` and the abort latch are
shared with `unitree/g1/controller` (verified button-by-button on the G1). The only Go2 difference is the
`LowState_` IDL: `unitree_sdk2py.idl.unitree_go.msg.dds_` (the G1 uses `unitree_hg`).

## The abort latch

`Go2Remote` **arms** only once the buttons are seen released (so a button held at start-up doesn't trip
it), then latches `aborted() == True` on the first press. The closed-loop locomotion helpers poll a
registered abort source every tick:

```python
loco.set_abort_source(remote.aborted)   # walk_* / turn_* return "aborted" the instant a button is hit
```

`reset()` clears the latch and re-disarms; `wait_until_armed()` blocks until the remote is released so a
residual press doesn't trip a fresh routine.

## Try it

```bash
# read-only, no motion — press buttons and watch ABORTED latch
python3 unitree/go2/controller/go2_remote.py --iface eth0 --seconds 30
```
