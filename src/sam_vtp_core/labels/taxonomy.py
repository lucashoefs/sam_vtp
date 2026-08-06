"""Global, platform-grounded traversability taxonomy.

Why this exists
---------------
The original `traversability_mappings.TRAVERSABILITY_MAPPINGS` assigns a
traversability value per (environment, class) pair, tuned per scene. Auditing it
across all 25 environments shows the scale is *environment-relative* rather than
absolute:

    ground   spans 0.70 - 1.00 across 20 environments
    floor    spans 0.70 - 1.00 across 12 environments
    z        spans 0.00 - 1.00 across  6 environments

so a flat concrete mine floor scores 0.70 while a city road scores 1.00. Two
consequences follow. First, part of the target is not predictable from the image
at all - the model must infer *which environment it is in* to know which offset
to apply - which shows up as error concentrated in large homogeneous regions
(measured: 62% of total error lies more than 16 px from any label boundary).
Second, the label no longer means anything physical, so a deployment threshold
chosen in one scene does not transfer to another.

This module replaces that with a single mapping from surface *physics* to a
score, shared by every environment.

Reference platform
------------------
Values are defined for a mid-size wheeled/tracked UGV:

    wheel/track radius      ~0.15 m
    ground clearance        ~0.20 m
    stable slope limit      ~30 deg  (marginal to 35 deg)
    max negotiable step     ~0.15 m

The score is "expected cost-free progress" on that surface:

    1.00  engineered hard flat surface, no penalty
    0.85  natural but compacted, or hard with joints//irregularity
    0.70  loose or vegetated but reliably drivable
    0.50  marginal: passable slowly or with risk
    0.30  strongly discouraged, only if no alternative
    0.00  not traversable

Anything not listed resolves through `KEYWORD_RULES` and finally to 0.0. That
default is deliberate and was verified safe: the unlisted classes are
overwhelmingly sky / wall / building / tree / vegetation-canopy.

Known ambiguity: `z` is an unnamed catch-all bucket that different environments
labelled anywhere from 0.0 to 1.0. It is treated here as a ground-like surface
and left to the geometric modulation to constrain, since it is elevated or steep
wherever it is not actually ground.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tier constants - use these rather than bare literals so the scale stays
# self-consistent when classes are added.
# ---------------------------------------------------------------------------
ENGINEERED = 1.00   # asphalt, concrete, tiled or panelled floor
HARD_JOINTED = 0.95  # hard surface with seams, plates, covers
COMPACTED = 0.90    # bare earth, packed dirt, solid timber decking
FIRM_NATURAL = 0.80  # gravel, packed stones, thin ground litter
VEGETATED = 0.70    # short grass, low ground plants
LOOSE_DEBRIS = 0.55  # scattered rubble, paper, cloth, boards
DENSE_LOW_VEG = 0.40  # bushes, hedges, dense foliage clumps
MARGINAL = 0.35     # stairs and steps: platform-dependent
BLOCKED = 0.00

# ---------------------------------------------------------------------------
# Explicit per-class assignment. Applies to every environment.
# ---------------------------------------------------------------------------
GLOBAL_TRAVERSABILITY = {
    # -- engineered hard flat -------------------------------------------
    'road': ENGINEERED,
    'roadway': ENGINEERED,
    'asphalt': ENGINEERED,
    'sidewalk': ENGINEERED,
    'pavement': ENGINEERED,
    'groundpavement': ENGINEERED,
    # A bridge deck is an engineered driving surface. Its absence resolved
    # through the BLOCKED default, which scored the ground under the robot as
    # an obstacle across NordicHarbor and MiddleEast -- in train, val and test
    # alike. Structural parts of the same class (railings, girders) are still
    # rejected: they match the hard-negative keywords first, and anything that
    # slips through stands proud of the deck and is cut by the height gate.
    'bridge': ENGINEERED,
    'bridgedeck': ENGINEERED,
    'groundpavers': ENGINEERED,
    'driveway': ENGINEERED,
    'walkway': ENGINEERED,
    'floor': ENGINEERED,
    'concretefloor': ENGINEERED,
    'concretefloorslab': ENGINEERED,
    'floorpanel': ENGINEERED,
    'floortile': ENGINEERED,
    'elevator': ENGINEERED,
    'tiledfloor': ENGINEERED,

    # -- hard, jointed or plated ----------------------------------------
    'manholecover': HARD_JOINTED,
    'stonefloor': HARD_JOINTED,
    'groundcobblestone': HARD_JOINTED,
    'drain': HARD_JOINTED,
    'pipecover': HARD_JOINTED,
    'metalfloor': HARD_JOINTED,
    'metalplatform': HARD_JOINTED,
    'steelplatform': HARD_JOINTED,
    'loadplatform': HARD_JOINTED,
    'platform': HARD_JOINTED,
    'metalramp': HARD_JOINTED,
    'ramp': HARD_JOINTED,

    # -- compacted natural / solid decking -------------------------------
    'ground': COMPACTED,
    'z': COMPACTED,              # catch-all bucket; geometry constrains it
    'landscape': COMPACTED,
    'groundtiretracks': COMPACTED,
    'dirt': COMPACTED,
    'soil': COMPACTED,
    'raisedwoodenfloor': COMPACTED,
    'woodenfloor': COMPACTED,
    'groundplank': COMPACTED,
    'groundplanks': COMPACTED,
    'woodenplatform': COMPACTED,
    'woodenwalkway': COMPACTED,
    'plank': COMPACTED,

    # -- firm but loose-surfaced -----------------------------------------
    'gravel': FIRM_NATURAL,
    'groundstones': FIRM_NATURAL,
    'groundembankment': FIRM_NATURAL,
    'roadmarkers': FIRM_NATURAL,
    'storesignsground': FIRM_NATURAL,
    'groundlight': FIRM_NATURAL,
    'tramlines': FIRM_NATURAL,
    'railway': FIRM_NATURAL,
    'rail': FIRM_NATURAL,
    'leafdecal': FIRM_NATURAL,
    'leaves': FIRM_NATURAL,
    'straw': FIRM_NATURAL,
    'paint': FIRM_NATURAL,

    # -- vegetated -------------------------------------------------------
    'grass': VEGETATED,
    'grassclump': VEGETATED,
    'groundplant': VEGETATED,
    'brush': VEGETATED,
    'stone': VEGETATED,
    'stonebase': VEGETATED,

    # -- loose debris ----------------------------------------------------
    'asphaltdebris': LOOSE_DEBRIS,
    'concretedebris': LOOSE_DEBRIS,
    'stonedebris': LOOSE_DEBRIS,
    'buildingdebris': LOOSE_DEBRIS,
    'debrispile': LOOSE_DEBRIS,
    'garbage': LOOSE_DEBRIS,
    'paper': LOOSE_DEBRIS,
    'paperdebris': LOOSE_DEBRIS,
    'cigarettebutt': LOOSE_DEBRIS,
    'cupcrumpled': LOOSE_DEBRIS,
    'cloth': LOOSE_DEBRIS,
    'towel': LOOSE_DEBRIS,
    'tarp': LOOSE_DEBRIS,
    'hose': LOOSE_DEBRIS,
    'board': LOOSE_DEBRIS,
    'woodboard': LOOSE_DEBRIS,
    'woodenboard': LOOSE_DEBRIS,
    'plywood': LOOSE_DEBRIS,
    'scaffoldingboards': LOOSE_DEBRIS,
    'pallet': LOOSE_DEBRIS,
    'woodenpallet': LOOSE_DEBRIS,
    'asphaltroller': LOOSE_DEBRIS,

    # -- dense low vegetation --------------------------------------------
    'bush': DENSE_LOW_VEG,
    'hedge': DENSE_LOW_VEG,
    'plant': DENSE_LOW_VEG,
    'plants': DENSE_LOW_VEG,
    'instancedfoliageactor': DENSE_LOW_VEG,
    'compcactusplantsgrass': DENSE_LOW_VEG,
    'compgrassrocksplants': DENSE_LOW_VEG,
    'compplantsgrasspaperplank': DENSE_LOW_VEG,

    # -- marginal: steps -------------------------------------------------
    # Deliberately below 0.5: a wheeled platform cannot climb these, a tracked
    # or legged one can. This is exactly the conditional feasibility that a
    # binary label cannot express.
    'stairs': MARGINAL,
    'stair': MARGINAL,
    'staircase': MARGINAL,
    'steps': MARGINAL,
    'concretestairs': MARGINAL,
    'metalstair': MARGINAL,
    'spiralstairs': MARGINAL,
    'bleachers': MARGINAL,
    'wheelbarrow': BLOCKED,
}

# ---------------------------------------------------------------------------
# Keyword fallback, applied in order to class names not listed above.
# The first matching rule wins, so more specific patterns come first.
# `None` means "explicitly not traversable" and stops further matching.
# ---------------------------------------------------------------------------
KEYWORD_RULES = [
    # Hard negatives that would otherwise be caught by a ground/floor keyword.
    (('wall', 'ceiling', 'roof', 'building', 'window', 'door', 'sky',
      'tree', 'trunk', 'branch', 'canopy', 'fence', 'railing', 'pole',
      'pillar', 'column', 'sign', 'light', 'lamp', 'wire', 'cable',
      'chain', 'pipe', 'vehicle', 'car', 'truck', 'person', 'barrel',
      'container', 'crate', 'machine', 'crane', 'scaffold', 'ladder',
      'rock', 'cliff', 'boulder', 'water', 'river', 'lake', 'sea',
      # Stacked/mounted objects that would otherwise match a surface keyword
      # (e.g. "floorradiator" -> floor, "woodenplankspile" -> planks).
      'radiator', 'heater', 'bundle', 'pile', 'stack', 'shelf'), BLOCKED),
    # Positive surface keywords.
    (('floor', 'pavement', 'sidewalk', 'walkway', 'road', 'asphalt', 'bridge'), ENGINEERED),
    (('ground', 'terrain', 'dirt', 'soil', 'earth'), COMPACTED),
    (('deck', 'platform', 'planks'), HARD_JOINTED),
    (('gravel', 'pebble'), FIRM_NATURAL),
    (('grass', 'moss', 'turf'), VEGETATED),
    (('debris', 'rubble', 'litter'), LOOSE_DEBRIS),
    (('foliage', 'bush', 'shrub', 'hedge'), DENSE_LOW_VEG),
    (('stair', 'step'), MARGINAL),
]

# Classes whose traversability is a property of the *surface underfoot* and
# should therefore be modulated by local slope and roughness. Everything with a
# non-zero base score qualifies, which is the point: the previous pipeline
# applied geometry to only 12 of 25 environments and a handful of classes, so
# just 8.4% of labelled pixels ever received a geometry-derived value.
MODULATE_ALL_TRAVERSABLE = True


def lookup(class_name: str, default: float = BLOCKED) -> float:
    """Global traversability for a semantic class name."""
    if not class_name:
        return default
    name = class_name.strip().lower()

    if name in GLOBAL_TRAVERSABILITY:
        return GLOBAL_TRAVERSABILITY[name]

    for keywords, value in KEYWORD_RULES:
        if any(k in name for k in keywords):
            return value

    return default


def build_lut(name_map: dict, default: float = BLOCKED):
    """Build a 256-entry uint8 -> traversability LUT for one environment.

    Args:
        name_map: {class_name: pixel_value} from the environment's label JSON.
    """
    lut = np.full(256, default, dtype=np.float32)
    for class_name, pixel_value in name_map.items():
        lut[int(pixel_value) % 256] = lookup(class_name, default)
    return lut


__all__ = [
    'GLOBAL_TRAVERSABILITY', 'KEYWORD_RULES', 'lookup', 'build_lut',
    'ENGINEERED', 'HARD_JOINTED', 'COMPACTED', 'FIRM_NATURAL', 'VEGETATED',
    'LOOSE_DEBRIS', 'DENSE_LOW_VEG', 'MARGINAL', 'BLOCKED',
]
