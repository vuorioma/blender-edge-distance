# Edge Distance

Bakes a texture, laid out in an object's UV space, where every texel stores how far that point on the mesh surface is from the nearest hard edge.

## Language

**Edge Distance Map**:
The output texture: a UV-space image whose value at each texel is the normalized surface distance from that point to the nearest hard edge.
_Avoid_: Edge mask, distance texture, edge map

**Hard Edge**:
A mesh edge that counts as a distance source, chosen by the hard edge rule. The distance at a hard edge is zero.
_Avoid_: Sharp edge (that is one possible rule, not the concept), seam

**Hard Edge Rule**:
The user-chosen criterion that decides which edges are hard edges: edges marked Sharp (the default) or edges whose face angle exceeds a threshold.

**Boundary Edge**:
An edge with only one adjacent face, such as a hole rim or the border of an open mesh. Counts as a hard edge by default.
_Avoid_: Open edge, naked edge

**UV Seam**:
An edge where the UV layout is cut. It is never a hard edge, and surface distance flows across it as if it weren't there.
_Avoid_: Seam edge, cut

**Surface Distance**:
Distance measured along the mesh surface, never through empty space.
_Avoid_: Euclidean distance, straight-line distance

**Unreachable Region**:
A connected part of the surface that contains no hard edge, so no hard edge can be reached from it across the surface. It is filled with the peak value (1).
_Avoid_: Island (that word means a UV island), orphan part

**Peak Distance**:
The largest surface distance anywhere on the object. Values in the edge distance map are normalized so the peak distance becomes 1.
_Avoid_: Max distance, falloff
