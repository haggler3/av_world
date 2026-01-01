# Software and hardware in the loop

`av_world/sil.py` (simulator side) and `av_world/sil_client.py` (an example controller and a ten-line client library). Your controller drives
one or more of the simulator's AVs; the rest of the city (human drivers, pedestrians, the other AVs) keeps running as usual. The plugin sits
outside the core: it uses the policy hook every AV already has, the core never imports it, and the simulator's speed without it is unchanged.

```bash
av-world sil --area chicago --egos 1 --mode lockstep --port 5555 --seconds 60        # terminal 1: the simulator waits for a controller
av-world sil-example --port 5555                                                     # terminal 2: the example controller
```
```python
from av_world import sil
s = sil.Session("new_york", egos=4, mode="realtime", deadline_ms=50, fallback="brake", shield="idm", port=5555)
summary = s.run(seconds=120)          # waits for the controller to connect, then drives; returns latency, misses, ego crashes ...
```

## How it works

```
simulator process                                                     controller (another process, machine or device)
-----------------                                                     ---------------------------------------------------
build the city, warm it up with the bundled policy
compile the step (before the clock starts)
accept the controller's TCP connection; send hello (dt, observation names, command format)
every 0.2 s of simulated time:
    the jitted step reaches the AV policy hook
    io_callback -> _exchange(obs [N, 28], AV mask):
        egos = the first K AVs present (replaced when one leaves)
        send {"type": "step", "step": k, "egos": [{"id", "obs", speed, gaps, signal ...}]}  ------>  read the step
                                                                                                     u = your_controller(ego) for each ego
        wait for {"type": "cmd", "step": k, "u": {...}}               <------------------------------  send {"type": "cmd", "step": k, "u": {...}}
            lockstep: up to timeout_s;  realtime: up to deadline_ms
        a reply for an earlier step is late: discard it
        an ego without a command this step: fallback (bundled policy | brake | hold the last command)
        realtime: sleep until wall-clock time k * dt / speed; count an overrun if already past it
    the commands go through the simulator's safety filter (shield: idm envelope | cbf barrier | off) and move the egos
send {"type": "bye", "summary": ...}
```

The observation is the same 28 features the learned policy sees (names in `sil.OBS_NAMES`), plus the main ones in physical units for
controllers that do not want normalised features: `speed_mps`, `lead_gap_m`, `lead_speed_mps`, `stop_gap_m`, `speed_limit_mps`, `signal`. The
command is `[u0, u1]`: `u0` in [-1, 1] is the acceleration request (1 = +3 m/s², -1 = -8 m/s²), `u1` the gap acceptance at the next junction
(0 = default).

## Modes

| | lockstep (SIL) | realtime (HIL) |
|---|---|---|
| clock | simulated time; the simulator waits for every command | wall clock (x `speed`) |
| a slow controller | slows the run down, never changes the result | misses its deadline: the ego gets the fallback, the late answer is discarded |
| measures | the controller's decisions | decisions and timing: latency percentiles, misses, overruns |
| use | regression tests of a software stack, CI | an ECU or embedded board in the loop, timing faults |

## Measured (this container, 4 CPU cores, controller on the same machine; `results/sil_bench.json`)

| run (Chicago Loop, 30 simulated s) | wall time | real-time factor | commands answered | latency p50 / p99 | misses | ego driven |
|---|---|---|---|---|---|---|
| lockstep, 1 ego | 0.50 s | 60x | 150 / 150 | 0.25 / 0.48 ms | 0 | 257 m |
| lockstep, 8 egos | 0.54 s | 56x | 150 / 150 | 0.33 / 0.60 ms | 0 | 1,474 m |
| realtime 1x, fast controller | 30.0 s | 1.0x (0 overruns) | 150 / 150 | 0.36 / 0.48 ms | 0 | 257 m |
| realtime 1x, controller takes 80 ms, deadline 50 ms, fallback brake | 30.0 s | 1.0x | 0 / 150 (149 late, discarded) | - | 150 | 6 m |

The last row is what HIL is for: a controller that is correct but too slow never gets to drive, and the run says so.

## Hardware in the loop in practice

* **Network.** The protocol is newline-delimited JSON over TCP, so a board on the LAN can connect directly (`--host 0.0.0.0` on the
  simulator side, the board connects to it). For CAN, serial or ROS 2, run a thin gateway next to the simulator that speaks this protocol on one
  side and the bus on the other; `sil_client.serve` is the loop it needs.
* **Timing.** Set `deadline_ms` to your control period's budget, `fallback` to what your safety concept says happens on a missed cycle, and
  `speed` to 1.0. Overruns mean the simulator itself could not keep up (a big city with many copies on a slow machine): use one copy, fewer egos,
  or lockstep.
* **Recording.** `av-world sil ... --record session.jsonl` writes every exchange (what was sent, what came back) for replay and debugging.

## Limits

* One controller connection per session, one copy of the city (the simulator is batched; SIL runs one copy so wall-clock time means something).
* The controller commands acceleration and junction gap acceptance, like the learned policy; lane changes and routes stay the simulator's.
* JSON over TCP costs about 0.3 ms per exchange locally; it is not a hard real-time transport. The pacing is `time.sleep`, good to about a
  millisecond on a normal OS, not a real-time kernel.
* Tested with controllers on the same machine (`tests/test_sil.py`: lockstep, a controller that really drives the ego, deadline misses and
  fallback); not tested with physical hardware here.
