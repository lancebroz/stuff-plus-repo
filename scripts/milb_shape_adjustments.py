#!/usr/bin/env python3
"""
milb_shape_adjustments.py
=========================
Two independent, multiplicative corrections to MiLB pitch *movement* (induced
vertical break VM and horizontal break HM). Velocity and spin are measured at
release and are never touched.

  1. BALL ADJUSTMENT (below Triple-A only)
     The minor-league ball differs from the MLB ball in drag/seam behavior, so
     shapes below AAA come in distorted relative to the big-league ball. A
     spin-efficiency-based, per-pitch-type correction maps them toward
     MLB-ball-equivalent movement. Triple-A already uses the MLB ball, so it is
     left raw. Applied per pitch.

  2. ALTITUDE ADJUSTMENT (all levels)
     Thinner air at elevation weakens the Magnus force, so break shrinks at high
     parks. Each pitcher gets one break-inflation factor = (reference air
     density) / (his average game-environment air density), modeled from his
     org + level (home park) and a 50/50 home/road split. Applied per pitcher.

The two corrections are both scalar multipliers on VM/HM and therefore COMMUTE;
the deployed pipeline applies altitude first, then the ball adjustment.

This file has NO third-party dependencies (pure Python). Drop it into any
project and import the functions you need.

Input conventions
-----------------
- VM : induced vertical break, GRAVITY REMOVED (so the density factor applies
       to it directly). HM : horizontal break. Units are inches in the source
       data but the corrections are unitless ratios, so any consistent unit works.
- spin_eff : spin efficiency as a WHOLE percent (e.g. 96.0, not 0.96).
- org : 3-letter club code (ANA=Angels, LA=Dodgers, OAK=Athletics, WAS=Nationals,
        CWS=White Sox, ...). "XX"/empty -> no altitude effect.
- level : one of "Triple-A", "Double-A", "High-A", "Single-A", "Rookie".
- ip_levels : iterable of (level_name, innings) for a pitcher across the season.
"""

# =============================================================================
# 1) BALL ADJUSTMENT
# =============================================================================

# expected = (A + B * SpinEff) * value ; SpinEff is a WHOLE percent.
# Sweeper ("ST") shares Slider ("SL") coefficients. Splitter / unknown types are
# not listed and pass through unchanged.
BALL_COEF = {  # code: {"VM": (A, B), "HM": (A, B)}
    "FF": {"VM": (1.005, -0.001), "HM": (0.946, -0.002)},
    "SI": {"VM": (0.939, -0.001), "HM": (0.965, -0.002)},
    "SL": {"VM": (0.981, -0.004), "HM": (0.902, -0.003)},
    "ST": {"VM": (0.981, -0.004), "HM": (0.902, -0.003)},  # = slider
    "FC": {"VM": (0.996, -0.002), "HM": (0.914, -0.003)},
    "CH": {"VM": (0.978, -0.001), "HM": (0.965, -0.001)},
    "CU": {"VM": (1.014, -0.001), "HM": (0.939, -0.003)},
}

# Only this fraction of the full modeled shift is applied (a deliberate
# dampening — see docs). 0.40 is the deployed value.
DAMPEN_DEFAULT = 0.40

# Pitch-type label -> code. Accepts either form in ball_adjust().
TYPE_MAP = {"Fastball": "FF", "Sinker": "SI", "Slider": "SL", "Sweeper": "ST",
            "Cutter": "FC", "Curveball": "CU", "Changeup": "CH", "Splitter": "FS"}

# Levels that use the MLB ball (no ball adjustment) vs. levels that get it.
NO_ADJUST_LEVELS = {"Triple-A", "AAA", "MLB", "Majors", "Major League"}
BALL_ADJUST_LEVELS = {"Double-A", "AA", "High-A", "A+", "Single-A", "A", "Rookie", "Rk"}


def _is_nan(x):
    return x is None or x != x


def is_below_aaa(level):
    """True if `level` uses the minor-league ball (i.e. gets the ball adjustment).
    Returns False for unknown levels; the deployed scorer instead defaults
    UNMATCHED pitchers to below-AAA (treat-as-adjust) out of caution."""
    return level in BALL_ADJUST_LEVELS


def ball_adjust(pitch, vm, hm, spin_eff, below_aaa, dampen=DAMPEN_DEFAULT):
    """Spin-efficiency ball adjustment on (VM, HM) -> (VM', HM').

    `pitch` may be a label ("Slider") or a code ("SL"). Returns the inputs
    unchanged if `below_aaa` is False, the pitch type has no coefficients
    (e.g. Splitter), or `spin_eff` is missing.
    """
    code = TYPE_MAP.get(pitch, pitch)
    if (not below_aaa) or code not in BALL_COEF or _is_nan(spin_eff):
        return vm, hm
    out = []
    for axis, val in (("VM", vm), ("HM", hm)):
        A, B = BALL_COEF[code][axis]
        expected = (A + B * spin_eff) * val
        out.append(val + dampen * (expected - val))
    return out[0], out[1]


# =============================================================================
# 2) ALTITUDE ADJUSTMENT
# =============================================================================

REF_ELEV_FT = 650.0  # reference (~MLB IP-weighted avg). Tunable. 0 = pure sea level.

# Park elevations (ft). Full PCL (AAA) + Texas League (AA) rosters are included so
# league road averages are exact; plus the elevated A+/A/complex parks. Anything
# not represented resolves to the reference (~no-op). Approximate by design.
PARK_ELEV = {
    # AAA Pacific Coast League (10)
    "ABQ": 5100, "RENO": 4505, "SLC": 4226, "ELP": 3740, "LV": 2030,
    "OKC": 1200, "RR": 720, "TAC": 250, "SAC": 30, "SUG": 80,
    # AA Texas League (10)
    "AMA": 3605, "MID": 2780, "SPR": 1300, "WIC": 1300, "NWA": 1300,
    "TUL": 700, "FRI": 650, "SA": 650, "ARK": 270, "CC": 35,
    # A+ elevated (Northwest League)
    "SPO": 1890,
    # A elevated (California League)
    "IE": 1070, "RC": 1200, "LE": 1300,
    # Rookie complexes
    "ACL": 1150, "FCL": 10,
}

# (org, level) -> home-park key. Anything not listed -> reference (no-op).
AFFIL = {
    # AAA / PCL
    ("COL", "Triple-A"): "ABQ", ("ARI", "Triple-A"): "RENO", ("ANA", "Triple-A"): "SLC",
    ("SD", "Triple-A"): "ELP", ("OAK", "Triple-A"): "LV", ("LA", "Triple-A"): "OKC",
    ("TEX", "Triple-A"): "RR", ("SEA", "Triple-A"): "TAC", ("SF", "Triple-A"): "SAC",
    ("HOU", "Triple-A"): "SUG",
    # AA / Texas League
    ("ARI", "Double-A"): "AMA", ("OAK", "Double-A"): "MID", ("STL", "Double-A"): "SPR",
    ("MIN", "Double-A"): "WIC", ("KC", "Double-A"): "NWA", ("LA", "Double-A"): "TUL",
    ("TEX", "Double-A"): "FRI", ("SD", "Double-A"): "SA", ("SEA", "Double-A"): "ARK",
    ("HOU", "Double-A"): "CC",
    # A+ elevated
    ("COL", "High-A"): "SPO",
    # A elevated (Cal League)
    ("ANA", "Single-A"): "IE", ("LA", "Single-A"): "RC", ("SD", "Single-A"): "LE",
}

# Cactus League (Arizona) orgs -> ACL (~1,150 ft) at Rookie; the rest -> FCL (~0).
CACTUS = {"ARI", "CHC", "CLE", "COL", "CWS", "CIN", "KC", "LA",
          "MIL", "OAK", "SD", "SEA", "SF", "TEX", "ANA"}

# League rosters (park keys) for road averages. Only elevated leagues need these.
LEAGUE_PARKS = {
    "PCL": ["ABQ", "RENO", "SLC", "ELP", "LV", "OKC", "RR", "TAC", "SAC", "SUG"],
    "TL":  ["AMA", "MID", "SPR", "WIC", "NWA", "TUL", "FRI", "SA", "ARK", "CC"],
}


def _rel_rho(elev_ft):
    """Standard-atmosphere air-density ratio vs. sea level at elevation (ft)."""
    return (1.0 - 6.8756e-6 * float(elev_ft)) ** 4.2559


_REF_RHO = _rel_rho(REF_ELEV_FT)


def _league(org, level):
    key = AFFIL.get((org, level))
    if level == "Triple-A" and key in LEAGUE_PARKS["PCL"]:
        return "PCL"
    if level == "Double-A" and key in LEAGUE_PARKS["TL"]:
        return "TL"
    return None


def home_elev(org, level):
    """Home-park elevation (ft) for an (org, level); reference if not tabulated."""
    key = AFFIL.get((org, level))
    if key:
        return PARK_ELEV[key]
    if level == "Rookie":
        return PARK_ELEV["ACL"] if org in CACTUS else PARK_ELEV["FCL"]
    return REF_ELEV_FT


def _road_rel_rho(org, level):
    lg = _league(org, level)
    if lg:
        home = AFFIL.get((org, level))
        others = [p for p in LEAGUE_PARKS[lg] if p != home]
        return sum(_rel_rho(PARK_ELEV[p]) for p in others) / len(others)
    if level == "Rookie":
        return _rel_rho(home_elev(org, level))  # complex ball: home==road metro
    return _REF_RHO


def altitude_factor_for_stint(org, level):
    """Break-inflation factor neutralizing ONE (org, level) stint to the reference.
    0.5*home + 0.5*league-road density. Returns 1.0 for XX/empty."""
    if not org or org == "XX" or not level:
        return 1.0
    env_rho = 0.5 * _rel_rho(home_elev(org, level)) + 0.5 * _road_rel_rho(org, level)
    return _REF_RHO / env_rho


def pitcher_altitude_factor(org, ip_levels):
    """IP-weighted break-inflation factor across a pitcher's levels.
    ip_levels: iterable of (level_name, innings). Returns 1.0 if empty/unknown."""
    if not org or org == "XX":
        return 1.0
    num = den = 0.0
    for lvl, ip in (ip_levels or []):
        if ip and ip > 0:
            num += altitude_factor_for_stint(org, lvl) * ip
            den += ip
    return num / den if den > 0 else 1.0


# =============================================================================
# 3) COMBINED
# =============================================================================

def adjust_shape(vm, hm, spin_eff, *, pitch, below_aaa,
                 alt_factor=None, org=None, level=None, ip_levels=None,
                 dampen=DAMPEN_DEFAULT):
    """Apply altitude (all levels) THEN the ball adjustment (below AAA) to (VM, HM).

    Supply the altitude factor directly via `alt_factor`, OR `org`+`ip_levels`
    (IP-weighted), OR `org`+`level` (single stint). Velocity and spin are never
    adjusted. Returns (VM', HM').
    """
    if alt_factor is None:
        if ip_levels is not None:
            alt_factor = pitcher_altitude_factor(org, ip_levels)
        elif level is not None:
            alt_factor = altitude_factor_for_stint(org, level)
        else:
            alt_factor = 1.0
    return ball_adjust(pitch, vm * alt_factor, hm * alt_factor, spin_eff, below_aaa, dampen)


if __name__ == "__main__":
    print("BALL ADJUSTMENT (below AAA)")
    for pt, se, vm, hm in [("Fastball", 95.0, 16.0, 9.0), ("Slider", 40.0, 2.0, -7.0)]:
        avm, ahm = ball_adjust(pt, vm, hm, se, below_aaa=True)
        print(f"  {pt:<9} SpinEff={se:>4}  VM {vm:>5.1f}->{avm:>5.2f}   HM {hm:>5.1f}->{ahm:>5.2f}")
    print("  (Triple-A is left raw: ball_adjust(..., below_aaa=False) returns inputs)\n")

    print(f"ALTITUDE FACTORS  (REF_ELEV_FT={REF_ELEV_FT:.0f}, ref density {_REF_RHO:.4f})")
    for (org, lvl), key in sorted(AFFIL.items(), key=lambda kv: -PARK_ELEV[kv[1]]):
        print(f"  {org:<3} {lvl:<9} {key:<4} {PARK_ELEV[key]:>5} ft   factor {altitude_factor_for_stint(org, lvl):.4f}")

    print("\nCOMBINED example — ARI Triple-A (Reno) curveball, full season at AAA:")
    f = pitcher_altitude_factor("ARI", [("Triple-A", 130.0)])
    out = adjust_shape(8.0, -6.0, 80.0, pitch="Curveball", below_aaa=False,
                       org="ARI", ip_levels=[("Triple-A", 130.0)])
    print(f"  altitude factor {f:.4f} (AAA: no ball adj)  ->  VM 8.0->{out[0]:.2f}  HM -6.0->{out[1]:.2f}")
