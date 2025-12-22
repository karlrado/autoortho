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

### Zoom Level (Critical Performance Factor)

The **Max Zoom Level** setting has the most significant impact on loading times and resource usage. Understanding this is crucial for optimizing performance.

#### How Zoom Levels Work

Each zoom level doubles the resolution in both dimensions, meaning **each zoom level increase requires 4× the resources** of the previous level:

| Zoom Level | Chunks per Tile | Relative Resources | Typical Use |
|------------|-----------------|-------------------|-------------|
| ZL14 | 16 (4×4) | 1× (baseline) | Low detail, fast loading |
| ZL15 | 64 (8×8) | 4× | Medium detail |
| ZL16 | 256 (16×16) | 16× | High detail (default) |
| ZL17 | 1024 (32×32) | 64× | Very high detail |
| ZL18 | 4096 (64×64) | 256× | Maximum detail, slowest |

#### Impact on Performance

**Downloads:** At ZL17, you need to download 4× more image chunks than ZL16. At ZL18, that's 16× more than ZL16.

**Processing:** Each chunk must be decoded (JPEG→RGB) and compressed (RGB→DDS). More chunks = more CPU time.

**Memory:** Higher zoom levels require more RAM to hold all the image data during processing.

**Storage:** Cache size grows exponentially with zoom level.

#### Recommendations by System

| System Type | Recommended Max Zoom | Notes |
|-------------|---------------------|-------|
| Low-end / Slow internet | ZL15 | Fastest loading, acceptable quality |
| Mid-range | ZL16 | Good balance (default) |
| High-end / Fast internet | ZL17 | High quality, longer loading |
| Enthusiast | ZL18 | Maximum quality, expect long loads |

#### Example: Loading Time Impact

Assuming 50ms average per chunk download:

| Zoom Level | Chunks | Theoretical Min Time |
|------------|--------|---------------------|
| ZL15 | 64 | ~3.2 seconds |
| ZL16 | 256 | ~12.8 seconds |
| ZL17 | 1024 | ~51.2 seconds |
| ZL18 | 4096 | ~3.4 minutes |

**Note:** Actual times vary based on parallelism, caching, and network conditions. These are theoretical minimums to illustrate the exponential scaling.

#### Config File Setting

```ini
# Maximum zoom level for imagery (14-18)
# Higher = better quality but exponentially longer loading times
# Each level increase = 4× more resources needed
maptype_override_zoom = 16
```

**Tip:** If you're experiencing long loading times or frequent missing tiles, consider lowering your max zoom level before adjusting other settings.

---

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

The maximum wall-clock time for a **complete tile** (all mipmap levels combined). This budget measures **active processing time only** - queue wait time doesn't count. The budget starts when AutoOrtho actually begins processing the tile's chunks, not when the tile is first requested.

| Value | Use Case | Effect |
|-------|----------|--------|
| 3.0 - 5.0s | Fast/smooth experience | Quicker loading, but more partial tiles |
| 5.0 - 15.0s | Balanced | Good quality with reasonable loading times |
| 15.0 - 30.0s | Maximum quality | Complete tiles, but longer loading times |

**How it works:**
1. X-Plane requests a tile from AutoOrtho (tile enters processing queue)
2. **Budget timer starts** when AutoOrtho actually begins downloading/processing chunks for this tile
3. AutoOrtho builds all mipmaps (4 → 3 → 2 → 1 → 0) sharing this budget
4. After `tile_time_budget` seconds of active processing, AutoOrtho builds the DDS with whatever is complete
5. Any incomplete areas use the configured `missing_color`

**Note:** Queue wait time (when other tiles are being processed first) does NOT count against the budget. This ensures fair time allocation even when many tiles are requested simultaneously.

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

#### Extended Fallback Timeout (`fallback_timeout`)
- **Type:** Float (seconds)
- **Default:** 3.0
- **Range:** 1.0 - 10.0 seconds
- **Config file:** `fallback_timeout = 3.0`

**Only applies when `fallback_extends_budget = True`.**

When extended fallbacks are enabled, this controls how long each lower-detail mipmap level waits for its chunks to download. The total additional time is this value multiplied by the number of mipmap levels tried (typically 3-4 levels).

| Value | Per-Level Wait | Total Extra Time (4 levels) | Use Case |
|-------|----------------|----------------------------|----------|
| 1.5s | 1.5 seconds | ~6 seconds | Fast - minimize extra wait |
| 3.0s | 3.0 seconds | ~12 seconds | Balanced (default) |
| 5.0s | 5.0 seconds | ~20 seconds | Quality - more time for slow connections |
| 10.0s | 10.0 seconds | ~40 seconds | Maximum - ensure fallbacks succeed |

**Example calculation:**
- `tile_time_budget = 10s` (exhausted after 10 seconds)
- `fallback_timeout = 3.0s`
- Fallback tries mipmap levels 1, 2, 3, 4 → 4 levels × 3.0s = 12 seconds max
- **Total worst-case time:** 10s + 12s = **22 seconds**

**Recommendation:** Start with 3.0s. If you see fallbacks timing out (check logs), increase to 5.0s. If loading is too slow, decrease to 1.5s.

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
- **Type:** Integer (minutes)
- **Default:** 10
- **Range:** 1 - 60 minutes
- **Config file:** `prefetch_lookahead = 10`

How far ahead (in minutes of flight time) to prefetch tiles.

| Value | At 150 kts | At 300 kts | At 500 kts | Use Case |
|-------|-----------|-----------|-----------|----------|
| 5 min | ~12nm | ~25nm | ~42nm | Conservative, less bandwidth |
| 10 min | ~25nm | ~50nm | ~83nm | Balanced (default) |
| 20 min | ~50nm | ~100nm | ~166nm | Longer flights, faster aircraft |
| 30 min | ~75nm | ~150nm | ~250nm | Cross-country flights |
| 60 min | ~150nm | ~300nm | ~500nm | Maximum prefetch coverage |

**Example:** At 300 knots ground speed with 10 minute lookahead:
- Distance: 300 kts × 10 min = 50 nautical miles ahead
- AutoOrtho will prefetch tiles up to 50nm in front of you

**Recommendation:** 
- Short flights / GA: 5-10 minutes
- Medium flights / Jets: 10-20 minutes  
- Long haul / Fast jets: 20-30 minutes

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
STATS: {'tile_create_count': 45, 'tile_create_avg_s': 8.42, 'tile_create_avg_by_mm': {0: 12.5, 1: 3.2, 2: 1.1},
        'mm_count:0': 12, 'mm_count:1': 26, 'chunk_budget_skipped': 8782, 'chunk_missing_count': 394}
```

### Tile Creation Time Statistics (Key for Tuning)

These statistics help you determine the optimal `tile_time_budget`:

| Statistic | Meaning |
|-----------|---------|
| `tile_create_count` | Total number of tiles created |
| `tile_create_avg_s` | **Average tile creation time in seconds** (key metric!) |
| `tile_create_avg_ms` | Average tile creation time in milliseconds |
| `tile_create_avg_by_mm` | Average creation time by mipmap level |
| `tile_create_time_ms:N` | Total time spent creating mipmap level N |

**How to use these stats to tune `tile_time_budget`:**

1. Run AutoOrtho with a high `tile_time_budget` (e.g., 60 seconds) to measure actual creation times
2. Look at `tile_create_avg_s` to see how long tiles actually take
3. Set your `tile_time_budget` slightly above this average for optimal performance

**Example interpretation:**
```
tile_create_avg_s: 8.42
tile_create_avg_by_mm: {0: 12.5, 1: 3.2, 2: 1.1}
```
- Average tile takes 8.42 seconds to create
- Mipmap 0 (highest detail) averages 12.5 seconds
- Setting `tile_time_budget = 15` would allow most tiles to complete fully
- Setting `tile_time_budget = 8` would give faster loading but some incomplete tiles

### Other Statistics

| Statistic | Meaning |
|-----------|---------|
| `mm_count:N` | Successful mipmap builds at level N (0=highest detail) |
| `mm_compress_time_ms:N` | Time spent on DDS compression for mipmap level N |
| `chunk_budget_skipped` | Chunks skipped because time budget ran out |
| `chunk_missing_count` | Chunks that ended up with missing color (no fallback worked) |

**Healthy indicators:**
- `tile_create_avg_s` < `tile_time_budget` = tiles completing within budget
- High `mm_count:0` values = high-detail tiles completing successfully
- Low `chunk_missing_count` = fallbacks working well

**Warning signs:**
- `tile_create_avg_s` > `tile_time_budget` = budget too short, increase it
- High `chunk_budget_skipped` = many chunks timing out, increase budget
- High `chunk_missing_count` = fallbacks not covering gaps, enable more fallbacks

### Breaking Down Tile Creation Time

Each tile creation has two main phases:

1. **Download + Compose** (typically 60-80% of time)
   - Downloading 64-4096 image chunks (depending on zoom level)
   - Decoding JPEG data
   - Compositing into a single image

2. **Compression** (typically 20-40% of time)
   - Converting RGBA to DDS format
   - Generating mipmaps

The stats show both the total time and compression-only time, letting you identify bottlenecks:
- If compression time is high → CPU-bound, consider lowering zoom level
- If download time is high → Network-bound, check internet speed

---

## Time-Based Exclusion

AutoOrtho includes a time-based exclusion feature that allows you to automatically disable AutoOrtho scenery during specific time ranges in the simulator. This is useful for night flying when satellite imagery provides little benefit.

### How It Works

When time exclusion is active:

1. AutoOrtho monitors the simulator's local time via the `sim/time/local_time_sec` dataref
2. During the exclusion period, DSF files are hidden from X-Plane
3. X-Plane falls back to its default scenery (which often has better night lighting)
4. When the exclusion period ends, AutoOrtho scenery becomes available again

### Safety Features

The time exclusion system includes important safety features:

- **Active DSF Protection:** DSF files that are currently in use by X-Plane will NOT be hidden, even if the exclusion period starts. This prevents crashes or graphical glitches.
- **Gradual Transition:** Only new DSF requests are blocked during exclusion. Previously loaded scenery continues to work until X-Plane naturally releases it.

### Configuration

#### Enable Time Exclusion (`enabled`)
- **Type:** Boolean (True/False)
- **Default:** False
- **Config file:** `[time_exclusion]` section, `enabled = True`

Enable or disable the time-based exclusion feature.

#### Start Time (`start_time`)
- **Type:** String (HH:MM format)
- **Default:** 22:00
- **Config file:** `start_time = 22:00`

The time when the exclusion period begins (24-hour format). For example, "22:00" for 10 PM.

#### End Time (`end_time`)
- **Type:** String (HH:MM format)
- **Default:** 06:00
- **Config file:** `end_time = 06:00`

The time when the exclusion period ends (24-hour format). For example, "06:00" for 6 AM.

#### Default to Exclusion (`default_to_exclusion`)
- **Type:** Boolean (True/False)
- **Default:** False
- **Config file:** `default_to_exclusion = False`

Controls behavior when sim time is not yet available (before flight starts):

| Setting | Behavior |
|---------|----------|
| **False** (default) | AutoOrtho works normally until sim time confirms exclusion |
| **True** | Assume exclusion is active until sim time proves otherwise |

**When to enable:**
- You want night flights to start with default scenery from the very beginning
- You don't want any AutoOrtho scenery loaded before sim time is available

**When to keep disabled:**
- You prefer AutoOrtho to work normally during loading screens
- You only want exclusion to apply when sim time is confirmed

### Example Configuration

To disable AutoOrtho between 10 PM and 6 AM (night hours):

```ini
[time_exclusion]
enabled = True
start_time = 22:00
end_time = 06:00
default_to_exclusion = False
```

To ensure exclusion is active from the moment AutoOrtho starts (before sim time is available):

```ini
[time_exclusion]
enabled = True
start_time = 22:00
end_time = 06:00
default_to_exclusion = True
```

### Overnight Ranges

The system correctly handles overnight time ranges. For example, if you set:
- Start: 22:00 (10 PM)
- End: 06:00 (6 AM)

AutoOrtho will be disabled from 10 PM until 6 AM the next morning.

### UI Configuration

You can configure time exclusion in the AutoOrtho Settings tab:

1. Go to **Settings** tab
2. Find the **Time Exclusion Settings** group
3. Check **Enable time-based exclusion**
4. Set the **Start time** and **End time** in HH:MM format

### Use Cases

- **Night Flying:** Satellite imagery is often dark or less useful at night. Default X-Plane scenery may have better night lighting.
- **Performance Optimization:** Reduce network usage and CPU load during night hours when visual quality matters less.
- **Dawn/Dusk Flying:** Exclude twilight hours when satellite imagery transitions may look unrealistic.

---

## Troubleshooting

See the [FAQ](faq.md#missing-color-tiles) for common issues related to:
- Missing color (green) tiles
- Long loading times
- In-flight stuttering

