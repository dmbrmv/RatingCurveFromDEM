# Provisional roadmap

* For river gauge crop DEM from MeritDEM
* Fill DEM, calculate flow direction, flow accumulation
* Crete pseudo-river network
* Find branch of biggest river in extent, which lay closest to the point of interest
* Crop dem by endpoints of this branch (RiverDEM)
* Derive Q(h) from observations on river gauge
* From RiverDEM create synthetic Q(h). Adjust "resistant" coefficient to match initial Q(h) in best possible way
* Make this N times
* Find regional dependencies on "resistant" coefficient.