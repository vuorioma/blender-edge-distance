# Blender Edge Distance

A Blender script that bakes an **edge distance map** for a UV-unwrapped object: a texture where every texel stores how far that point on the surface is from the nearest hard edge. Hard edges are black (0), and the value rises to white (1) at the point farthest from any hard edge.

Distance is measured along the mesh surface, so it never cuts through empty space, and it flows smoothly across UV seams.

Requires Blender 5.2 or newer.

## Usage

1. Open the **Scripting** tab in Blender.
2. Select the object you want to bake. It must be a mesh with a UV map.
3. Click **New** in the Text Editor, paste in the contents of [`edge_distance.py`](edge_distance.py), and click **Run Script** (▶).

The result is an image named `<Object>_EdgeDistance`, which you can open in the Image Editor or use in a material through an Image Texture node. Running the script again overwrites the image in place, so materials using it update automatically.

> **The image isn't saved to disk.** Save it from the Image Editor with *Image > Save As*. Choose OpenEXR to keep full float precision.

## Settings

Edit the constants at the top of the script before running it:

| Setting | Default | Meaning |
|---|---|---|
| `RESOLUTION` | `1024` | Width and height of the output image in pixels |
| `HARD_EDGE_RULE` | `'SHARP'` | `'SHARP'` uses edges marked Sharp; `'ANGLE'` uses edges whose face angle exceeds `ANGLE_THRESHOLD` |
| `ANGLE_THRESHOLD` | `30.0` | Angle in degrees, used when `HARD_EDGE_RULE` is `'ANGLE'` |
| `BOUNDARY_EDGES_ARE_HARD` | `True` | Treat open edges (hole rims, borders of open meshes) as hard edges |
| `MARGIN` | `16` | Pixels of padding added around UV islands |

## Notes

- The **evaluated** mesh is baked: modifiers are applied, and distances are measured in world space, so unapplied scale is fine.
- The UV map highlighted in the object's UV Maps list is used.
- UV seams don't count as hard edges.
- Values are normalized, so the point farthest from any hard edge is always 1.
- A **Bevel** modifier replaces sharp edges with bevel faces. If the script reports that no hard edges were found, enable the modifier's *Mark Sharp* option or switch to the `'ANGLE'` rule.
- If UVs overlap, the script warns you and the overlapping areas will be wrong.
- Separate mesh parts with no hard edges are filled with 1, and the script warns about them.

See [`GLOSSARY.md`](GLOSSARY.md) for the terms used in the code.
