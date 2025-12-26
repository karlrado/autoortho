# Configuration

Setup and configuration for most use cases should be pretty simple. The important settings are:

* Scenery install path
* X-Plane install path
* Downloader directory

## Scenery install path
This is the location that scenery will be installed to.  Previously this defaulted to a user's existing X-Plane Custom Scenery directory, but that is no longer the case.  

It should be possible to set this to a convenient location with enough room to install scenery packages.  Each scenery package can take around 20-30GB.

This can be an external NAS or separate drive, but I have not tried all drive combinations.  Speed of external storage will naturally impact performance to a certain degree.  Plan accordingly.

## X-Plane install path
This is the X-Plane install location.  Under this directory should be X-Plane's `Custom Scenery` directory. 
*IT IS IMPORTANT THIS IS THE CORRECT LOCATION*

From this directory AutoOrtho will create mount points and run the program.

*IF THIS IS NOT CORRECT THINGS WILL NOT WORK RIGHT*

## Download directory
This is the path that will be used to temporarily store zip files and other fetched files for scenery setup.  By default this will be in under the user's home dir under `.autoortho-data/downloads`

For Windows users, it is highly recommended to set a Windows Defender exception to this directory otherwise expect *VERY* slow setup of scenery.

This folder can be set to any convenient location with enough space for scenery downloads (20-30GB per).

## User config file location

The configuration file `.autoortho` is located in the user's home directory.  

## Performance Tuning

AutoOrtho includes advanced performance settings that allow you to balance image quality against loading times and stuttering. These settings are available in the Settings tab under "Performance Tuning".

Key settings include:
- **Tile Time Budget** - Maximum time to wait for a tile before returning results
- **Fallback Level** - How aggressively to find replacement imagery for failed chunks
- **Spatial Prefetching** - Proactively download tiles ahead of your aircraft
- **Dynamic Zoom Levels** - Automatically adjust imagery quality based on altitude

For detailed configuration options and recommended settings for different use cases, see the [Performance Tuning Guide](performance.md).

For troubleshooting missing tiles or stuttering issues, see the [FAQ](faq.md#missing-color-tiles).

## Dynamic Zoom Levels

AutoOrtho can automatically adjust imagery zoom levels based on your altitude Above Ground Level (AGL). This provides:
- Higher detail imagery when flying low
- Lower detail (faster loading) imagery at high altitudes
- Terrain-aware calculations — flying at 10,000ft MSL over 5,000ft mountains uses higher quality than 10,000ft over ocean

Configure quality steps in **Settings** → **Imagery** → **Dynamic Zoom Mode**.

See the [Performance Tuning Guide](performance.md#dynamic-zoom-levels) for detailed configuration.

## SimBrief Integration

AutoOrtho can integrate with your SimBrief account to use your flight plan data for:
- **Dynamic Zoom**: Use planned altitudes at waypoints instead of velocity predictions
- **Prefetching**: Download tiles along your actual flight path ahead of time

To set up SimBrief integration:
1. Go to **Settings** → **Setup** tab
2. Enter your **SimBrief User ID**
3. Click **Fetch Flight Data** after filing your flight plan
4. Enable the toggle to use flight data for calculations
5. Optionally adjust the **Route Calculation Settings** that appear below the toggle

### Route Calculation Settings

When flight data is loaded and the "Use Flight Data" toggle is enabled, additional settings become available:

| Setting | Description |
|---------|-------------|
| **Route Consideration Radius** | How far (nm) to look for waypoints when calculating tile altitude. Larger values are more conservative. |
| **Route Deviation Threshold** | Maximum distance (nm) off-route before falling back to DataRef-based calculations. |
| **Route Prefetch Radius** | How far (nm) around waypoints to prefetch tiles. Larger values prefetch more tiles. |

> **ℹ Real-time Changes:** These settings take effect immediately when modified — no restart required. Use **Save Config** to persist your values for future sessions.

See the [Performance Tuning Guide](performance.md#simbrief-integration) for detailed information and limitations.

