# SWARM — a WebGPU compute-driven boids battler

A massive 3D space battle where **every ship is an individual agent**. Two equal
fleets — cyan and amber — spawn on opposite sides, charge each other, flock with
their own kind (Boids), seek and grind down the enemy, and die. Up to **200,000
agents**, with their entire lifecycle — flocking, neighbour search, combat,
death and reinforcement — running **inside GPU compute shaders**. The CPU never
touches a single agent's position.

A single self-contained `index.html` (Three.js r0.184, WebGPU + TSL), no build
step.

## Why WebGL couldn't do this

In WebGL there are no compute shaders and no storage buffers. Simulating 100k+
interacting agents means GPGPU hacks: encode positions into a floating-point
texture, draw a full-screen quad to "update" them, ping-pong render targets, and
either read the result back to the CPU or wrestle it through transform feedback.
Neighbour search across a swarm that big is effectively impossible at frame rate.

WebGPU removes the hack. General-purpose **compute pipelines** read and write
**storage buffers** directly, and those exact same buffers are bound straight
into the **render pipeline** — no CPU readback, no texture encoding, no quad.

## How it works

The whole entity array lives on the GPU as two `vec4` storage buffers:

| buffer | x | y | z | w |
|--------|---|---|---|---|
| `posBuf` | position.x | position.y | position.z | **health** (≤0 ⇒ dead) |
| `velBuf` | velocity.x | velocity.y | velocity.z | **faction** (0 cyan / 1 amber) |

Every frame runs three compute passes, then one instanced draw — all reading the
same buffers:

1. **`clearKernel`** — zero the per-cell counts of a uniform spatial grid
   (52³ ≈ 140k cells) and the two faction tallies.
2. **`scatterKernel`** — each live agent hashes its position to a grid cell and
   claims a slot with `atomicAdd`. This is the trick that makes neighbour search
   cheap: instead of comparing every agent to every other (O(n²) ≈ 10¹⁰ at 100k),
   each agent only looks at the ≤27 cells around it.
3. **`boidsKernel`** — the simulation. For each live agent, walk the 3×3×3
   neighbouring cells and:
   - **same faction** → classic Boids: separation, alignment, cohesion;
   - **enemy** → steer toward the nearest one, and take damage for every enemy
     inside kill range. Damage is applied to *yourself* based on how many
     enemies surround you — so it's race-free (no cross-agent writes) and
     produces emergent Lanchester-style attrition: get outnumbered and you melt.

   Velocity is integrated, speed-clamped, and a soft boundary keeps the brawl on
   screen. Dead agents optionally **reinforce** — respawn at their fleet's edge —
   so the front line is endlessly fed.

4. **Render** — one `InstancedMesh` of 200k little cone "darts". The vertex
   stage (`positionNode`) reads `posBuf`/`velBuf` by `instanceIndex`, builds an
   orthonormal basis to point each ship along its velocity, and collapses dead
   ships to a degenerate point. Faction drives colour; near-death ships flash
   white-hot. A TSL **bloom** pass makes the swarm glow.

The only thing that ever comes back to the CPU is a tiny **2-integer faction
count**, read back a few times a second for the HUD bars. That's it.

## Run

ES modules + a CDN import map need HTTP — you can't open `index.html` from
`file://`:

```bash
python3 -m http.server 8099
# then open http://localhost:8099/
```

No build, no dependencies to install. Three.js is pinned to `three@0.184.0` via
the import map.

**Requires WebGPU** — a recent Chrome / Edge, or Safari Technology Preview. If
WebGPU is unavailable the page says so instead of half-running.

## Controls

| Input | Action |
|-------|--------|
| **Drag** | rotate camera · **scroll** zoom |
| **FLOCKING** sliders | cohesion / alignment / separation / perception radius |
| **COMBAT** sliders | aggression (enemy seek), damage, max speed |
| **agents** slider | live agent count, 5k → 200k (reseeds the battle) |
| **❚❚** | pause the simulation (rendering continues) |
| **RESET** | reseed both fleets |
| **REINF** | toggle reinforcement — off = fight to extinction, one fleet wins |
| **BLOOM** | toggle the glow post-process |
| **ORBIT** | toggle slow auto-rotation of the camera |

## Performance

Measured on an Apple-Silicon GPU (GPU time = compute + draw per frame, the real
cost — FPS is vsync-capped):

| agents | GPU ms / frame |
|--------|----------------|
| 100,000 | ~2 ms (120 fps) |
| 200,000 | ~3.3 ms (70 fps) |

Cost scales with *live* agents — dead ones early-out of the compute pass — and
with crowding (the per-cell capacity bounds the inner neighbour loop, so even a
huge pile-up can't blow up the frame time).

## Architecture notes

- **Buffers are allocated once at the maximum (200k).** The `agents` slider just
  moves an `uActive` uniform and re-runs a GPU `seedKernel`; inactive agents are
  seeded dead and skipped. No reallocation, no CPU seeding — spawn positions and
  velocities are generated on the GPU from per-index hashes.
- **Faction is assigned by index parity**, not by index half. When a dense cell
  overflows its capacity and drops neighbours, parity keeps both factions
  equally affected — otherwise low-index agents win slots and one colour gets a
  systematic edge.
- **All combat writes are self-writes.** Reading neighbours is fine in parallel;
  the moment you let agents damage *each other* you have a data race. Counting
  surrounding enemies and damaging yourself sidesteps it entirely.

Built in the spirit of the sibling WebGPU demos (`webgpu_bloom`,
`terraine_demo`): single file, TSL compute, GPU-resident state, GPU-ms readout.
