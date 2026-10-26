# Moving railway comparison route

Open `moving_dataset/moving_railway.blend` in Blender 4.2 or newer. Press **Numpad 0** for the onboard view and **Space** to play the train journey. The route is 1,260 metres long; the train travels at 12 m/s over 105 seconds (2,521 animation frames at 24 fps). Select `EXTERIOR_FOLLOW` as the active camera to follow the four-car sensor train from outside. The active `RGB_CENTER`, `RGB_LEFT`, `RGB_RIGHT` cameras and `LIDAR_ORIGIN` are parented to the moving train.

The continuous route includes three running tracks, curved alignment, two platforms and station buildings, sidings, parked passenger trains and freight stock, signals and gantries, catenary poles, overhead contact and messenger wires, concrete sleepers, detailed rail profiles, ballast, fencing, service road, residential buildings and porous woodland. People have articulated limbs and clothes; road vehicles, rail vehicles and track hardware have additional component detail. These are procedural assets with textured materials and physical sky lighting.

Every OSDaR23 class has at least **12 independently placed assemblies**, with more infrastructure and people. `manifest.json` records exact counts. Groups such as crowd and group_of_bicycles count as one assembly containing multiple members. Scene counts exclude the moving ego train. Staged fires are beside the route on maintenance aprons. Scenes and classes are inspired by OSDaR23; they do not reproduce the recorded Hamburg sequences.

## Captures

`capture_route.py` exports 63 synchronized train poses spaced 20 m apart along the route:

- `captures/rgb/`: ordered onboard center-camera PNGs.
- `captures/lidar/`: compressed NPZ scans with `xyz`, `instance_id`, `class_id`; selected scans also in ASCII PLY.
- `captures/annotations/`: camera and sensor poses, intrinsics, route station, full world AABBs and clipped projected amodal boxes.
- `captures/validation.json`: actual observed unique LiDAR instances per class, distinct from scene counts.

The continuous animation is at 24fps. Exported sensor samples are every 1.667 seconds, not the original dataset's 10Hz sensor stream. RGB frame and LiDAR scan share the same animation time. Camera boxes are amodal projections, can include occluded objects, and are not pixel masks. No model-specific class mapping or official RailLabel compatibility is asserted. Use `manifest.json` to remap local zero-based class IDs to your model.

LiDAR is instantaneous idealized nearest-hit geometry: 64 elevation samples (-30 to +12 degrees), 1,536 azimuth samples (-90 to +90 degrees), 0.3–200 m range. XYZ is in the moving sensor frame (X forward, Y left, Z up); matrices map sensor coordinates to Blender world metres. Ego train self-returns have instance 0 and class -1, like other unlabeled background. Labels track target assemblies consistently across time. Smoke/flame use geometric visual proxies; their ray returns are not aerosol or thermal physics. There is no calibrated multi-LiDAR response, IR, radar, noise or motion distortion. The four-car ego train is rigid on the gentle curves rather than articulated bogie dynamics.

## Rebuild

From CROSS:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe' --background --factory-startup --threads 6 --python osdar23_synthetic/moving_route.py
& 'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe' --background --factory-startup --threads 6 --python osdar23_synthetic/refine_route.py
& 'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe' --background --factory-startup --threads 6 --python osdar23_synthetic/capture_route.py
& 'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe' --background --factory-startup --threads 6 --python osdar23_synthetic/verify_and_preview.py
```

`moving_route.py` reuses primitive/class helpers from `generate.py` and woodland geometry from `railway_lidar/foliage.py`. Keep those source files beside the project as supplied.

`sampled_journey.mp4` plays the 63 exported samples at 8fps as a fast route preview. The Blender file contains the full continuously interpolated 105-second motion. `verification.json` checks class/instance labels, finite scan coordinates, sensor range, matching capture times, RGB files, target counts, train displacement and the sensor attachments.
