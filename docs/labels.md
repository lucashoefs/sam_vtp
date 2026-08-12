# Label generation

<p align="center"><img src="label-pipeline.png" width="100%" alt="Traversability supervision pipeline"></p>

RGB, depth and semantics on the left are the inputs; height, slope and the final
target `y` on the right are what the pipeline derives. The target cannot be read
off either the semantics or the geometry alone, which is the reason for training
a model on it rather than computing it at runtime.

The training target is not hand-annotated. It is built per frame as

```
y = semantic_prior(class) x slope_factor x roughness_factor x height_gate
```

so semantics say what the surface is and geometry says what state it is in. You
need both: a concrete floor is 1.00 flat and 0.30 at 30 degrees, and grass,
bushes, trunks and canopy all share one semantic class in the forest
environments.

The semantic prior is absolute rather than per-environment, so a threshold chosen
in one scene means the same thing in another.

```bash
sam-vtp-labels --root /data/tartanground --workers 24
```

Writes `<traj>/trav_lcam_front/{frame:06d}_lcam_front_trav.png`, 8-bit, 255 being
fully traversable.

`labels/taxonomy.py` holds the class-to-score mapping and the reference platform
it is grounded in (about 0.15 m wheel radius, 0.20 m clearance, 30 degree stable
slope, 0.15 m step). `labels/generate.py` holds the geometry. Both docstrings
record why each constant is what it is, so read them before changing one.

## Four choices that shape the output

**Camera height is estimated once per trajectory**, not per frame. It is a
rigid-body constant, and a per-frame ground-pixel median swings by half a metre
on vegetated terrain, which moves the height gate around.

**Slope is measured at terrain scale** with an 11 px stencil. Depth resolves
individual leaves, so a narrow stencil measures the tilt of whichever leaf a
pixel landed on instead of the grade of the ground.

**Modulation is smoothed at sigma 12 and normalised by the traversable mask.**
The decoder predicts at stride 4, so finer detail cannot be represented anyway.
The mask normalisation stops a wall, which is legitimately near-vertical, from
bleeding a dark halo onto the ground next to it. Class boundaries stay sharp
either way, since only the modulation is smoothed and never the prior.

**Height and slope are gravity-aligned** using per-frame pose. Treating the
optical axis as horizontal makes the height error grow linearly with range (around 0.87 m at 50 m per degree of pitch) so distant flat ground drifts up
until it crosses the gate and gets labelled an obstacle. With no pose file the
generator falls back to the level-camera assumption and tells you.
