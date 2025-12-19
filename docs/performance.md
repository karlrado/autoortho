# Performance Tuning Guide

AutoOrtho includes advanced performance tuning options that allow you to balance image quality against loading times and in-flight stuttering. This guide explains each setting and provides recommendations for different use cases.

## Quick Start: Performance Presets

For most users, the easiest way to configure performance is using the **Performance Preset** dropdown in Settings → Performance Tuning:

| Preset | Best For | Trade-off |
|--------|----------|-----------|
| **Fast** | Weak CPUs, slow internet, stutter-free experience | May have occasional missing/low-res tiles |
| **Balanced** | Most users | Good balance of quality and performance |
| **Quality** | Fast CPUs, fast internet, maximum image quality | May have longer loading times |
| **Custom** | Advanced users who want fine-grained control | Manual configuration required |

---

## Detailed Settings Reference

### Time Budget System

The Time Budget system controls how long AutoOrtho spends loading each tile before returning a result to X-Plane.

#### Enable Time Budget (`use_time_budget`)
- **Type:** Boolean (True/False)
- **Default:** True
- **Config file:** `use_time_budget = True`

When enabled, AutoOrtho uses a wall-clock time limit for tile requests instead of the legacy per-chunk timeout system. This provides more predictable performance and reduces stuttering.

**Recommendation:** Keep enabled unless you experience issues.

---

#### Tile Time Budget (`tile_time_budget`)
- **Type:** Float (seconds)
- **Default:** 5.0
- **Range:** 0.5 - 30.0 seconds
- **Config file:** `tile_time_budget = 5.0`

The maximum wall-clock time AutoOrtho will spend loading a single tile. Each tile consists of up to 256 chunks (16×16 grid) that need to be downloaded, decoded, and compressed.

| Value | Use Case | Effect |
|-------|----------|--------|
| 3.0 - 5.0s | Fast/smooth experience | Less waiting, but some tiles may be incomplete |
| 5.0 - 15.0s | Balanced | Good quality with reasonable loading times |
| 15.0 - 30.0s | Maximum quality | Complete tiles, but longer loading times |

**How it works:**
1. X-Plane requests a tile from AutoOrtho
2. AutoOrtho starts downloading all 256 chunks in parallel
3. After `tile_time_budget` seconds, AutoOrtho returns whatever it has completed
4. Any incomplete chunks use fallback images (from cache, scaling, or lower detail)

**Note:** Each tile covers a large geographic area (approximately 1 square degree of latitude/longitude at zoom 16). Higher budgets allow more time for all chunks to download and process.

---

### Fallback System

When chunks fail to download in time, the fallback system provides alternative imagery to prevent missing tiles.

#### Fallback Level (`fallback_level`)
- **Type:** String (none, cache, full)
- **Default:** cache
- **Config file:** `fallback_level = cache`

Controls which fallback mechanisms are enabled:

| Level | Fallback 1 (Disk Cache) | Fallback 2 (Scale Mipmaps) | Fallback 3 (Network) | Use Case |
|-------|------------------------|---------------------------|---------------------|----------|
| **none** | ❌ Disabled | ❌ Disabled | ❌ Disabled | Fastest, may have green/missing tiles |
| **cache** | ✅ Enabled | ✅ Enabled | ❌ Disabled | Balanced - uses cached data only |
| **full** | ✅ Enabled | ✅ Enabled | ✅ Enabled | Best quality - can download lower-detail alternatives |

**Fallback Chain Explained:**

1. **Fallback 1 - Disk Cache:** Searches your local cache for a lower-zoom version of the same imagery. Fast and free.

2. **Fallback 2 - Scale from Mipmaps:** Scales imagery from already-built lower-detail mipmap levels. Very fast.

3. **Fallback 3 - Network Download:** Downloads lower-detail imagery on-demand from the server. Slowest but provides the best fill-in quality.

**Recommendation:** Use `cache` for most users. Use `full` if you have fast internet and prefer complete imagery over speed.

---

#### Fallback Extends Budget (`fallback_extends_budget`)
- **Type:** Boolean (True/False)
- **Default:** False
- **Config file:** `fallback_extends_budget = False`

**Only applies when `fallback_level = full`.**

When enabled, network fallbacks (Fallback 3) will continue even after the tile time budget is exhausted. This prioritizes image quality over strict timing.

| Setting | Behavior | Effect |
|---------|----------|--------|
| **False** (default) | Fallbacks respect budget | Faster loading, may have some missing tiles |
| **True** | Fallbacks ignore budget | Better quality, may cause longer load times |

**When to enable:**
- You prioritize having complete imagery over loading speed
- You have a fast, reliable internet connection
- You don't mind occasional longer loading times

**When to keep disabled:**
- You want predictable, stutter-free performance
- Your internet is slow or unreliable
- Loading speed is more important than perfect imagery

---

### Spatial Prefetching

The prefetching system proactively downloads tiles ahead of your aircraft to reduce in-flight stuttering.

#### Enable Prefetching (`prefetch_enabled`)
- **Type:** Boolean (True/False)
- **Default:** True
- **Config file:** `prefetch_enabled = True`

When enabled, AutoOrtho monitors your aircraft's position, heading, and speed to predict which tiles you'll need next and downloads them in advance.

**How it works:**
1. AutoOrtho tracks your aircraft's heading and ground speed
2. It calculates which tiles are in your flight path
3. It downloads those tiles in the background before you reach them
4. When X-Plane requests those tiles, they're already cached

**Recommendation:** Keep enabled for the smoothest in-flight experience.

---

#### Lookahead Time (`prefetch_lookahead`)
- **Type:** Integer (seconds)
- **Default:** 30
- **Range:** 10 - 120 seconds
- **Config file:** `prefetch_lookahead = 30`

How far ahead (in seconds of flight time) to prefetch tiles.

| Value | Use Case |
|-------|----------|
| 10-20s | Slow aircraft (GA, helicopters) |
| 30-60s | Medium speed (turboprops, regional jets) |
| 60-120s | Fast aircraft (jets, supersonic) |

**Example:** At 300 knots ground speed with 60 second lookahead:
- Distance: 300 kts × 60s ÷ 3600 = 5 nautical miles ahead
- AutoOrtho will prefetch tiles up to 5nm in front of you

---

### Legacy Settings

These settings are from the original timeout system and are still available for compatibility.

#### Max Wait (`maxwait`)
- **Type:** Float (seconds)
- **Default:** 1.5
- **Config file:** `maxwait = 1.5`

The per-chunk timeout used when the time budget system is disabled. Each chunk waits up to this long before timing out.

**Note:** When `use_time_budget = True`, this setting is largely superseded by `tile_time_budget`.

#### Suspend Max Wait at Startup (`suspend_maxwait`)
- **Type:** Boolean
- **Default:** True
- **Config file:** `suspend_maxwait = True`

When enabled, uses a much longer timeout during X-Plane startup to ensure initial tiles load completely before flight begins.

---

## Recommended Configurations

### Stutter-Free Flying (Prioritize Performance)
```ini
use_time_budget = True
tile_time_budget = 5.0
fallback_level = cache
fallback_extends_budget = False
prefetch_enabled = True
prefetch_lookahead = 45
```

### Maximum Quality (Prioritize Imagery)
```ini
use_time_budget = True
tile_time_budget = 20.0
fallback_level = full
fallback_extends_budget = True
prefetch_enabled = True
prefetch_lookahead = 60
```

### Slow Internet Connection
```ini
use_time_budget = True
tile_time_budget = 15.0
fallback_level = cache
fallback_extends_budget = False
prefetch_enabled = True
prefetch_lookahead = 90
```

### Weak CPU / Limited System
```ini
use_time_budget = True
tile_time_budget = 3.0
fallback_level = none
fallback_extends_budget = False
prefetch_enabled = False
```

---

## Understanding the Statistics

AutoOrtho logs performance statistics that can help you tune your settings:

```
STATS: {'chunk_miss': 1909597, 'req_ok': 80408, 'mm_counts': {0: 12, 1: 26, ...}, 
        'partial_mm_counts': {0: 2139}, 'chunk_budget_skipped': 8782, 'chunk_missing_count': 394}
```

| Statistic | Meaning |
|-----------|---------|
| `mm_counts` | Successful complete mipmap builds by level (0=highest detail) |
| `partial_mm_counts` | Partial builds (some chunks missing) by level |
| `chunk_budget_skipped` | Chunks skipped because time budget ran out |
| `chunk_missing_count` | Chunks that ended up with missing color (no fallback worked) |

**Healthy indicators:**
- High `mm_counts` values = tiles completing successfully
- Low `partial_mm_counts` = few partial tiles
- Low `chunk_missing_count` = fallbacks working well

**Warning signs:**
- High `partial_mm_counts[0]` = mipmap 0 (highest detail) frequently incomplete
- High `chunk_budget_skipped` = budget too short, consider increasing
- High `chunk_missing_count` = fallbacks not covering gaps, consider enabling more fallbacks

---

## Troubleshooting

See the [FAQ](faq.md#missing-color-tiles) for common issues related to:
- Missing color (green) tiles
- Long loading times
- In-flight stuttering

