"""Edge Distance Map baker.

Bakes a texture in the active object's UV space where every texel stores the
surface distance from that point to the nearest hard edge, normalized so the
peak distance is 1. See GLOSSARY.md for the terms used here.

Usage: open this file in Blender's Text Editor, make the target object active,
and press Run Script. The result is written to an image datablock named
"<Object>_EdgeDistance". It is not saved to disk: save it from the Image Editor.
"""

import math
import time

import numpy as np

try:
    import bpy
except ImportError:  # Allows the NumPy core to be imported outside Blender.
    bpy = None


# Settings ------------------------------------------------------------------

RESOLUTION = 1024
HARD_EDGE_RULE = 'SHARP'         # 'SHARP' (edges marked Sharp) or 'ANGLE'
ANGLE_THRESHOLD = 30.0           # Degrees, used when HARD_EDGE_RULE == 'ANGLE'
BOUNDARY_EDGES_ARE_HARD = True
MARGIN = 16                      # Pixels of padding around UV islands


# Internals -----------------------------------------------------------------

# Texels whose center lies within this many pixels of a triangle still get a
# surface position (extrapolated), so island borders and seams are covered.
DILATE_RADIUS = 0.75
# A texel-grid edge is rejected when its 3D length exceeds this multiple of the
# expected length, which keeps UV-adjacent but surface-distant texels apart.
EDGE_SLACK = 4.0
# Upper bound on the number of elements processed at once.
CHUNK = 1 << 22

NEIGHBORHOOD = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class EdgeDistanceError(Exception):
    pass


# Array helpers -------------------------------------------------------------

def expand(counts):
    """For runs of the given lengths, return (run index, index within run)."""
    run = np.repeat(np.arange(len(counts)), counts)
    local = np.arange(run.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return run, local


def ranges(starts, counts):
    """Concatenate the integer ranges [start, start + count)."""
    run, local = expand(counts)
    return starts[run] + local


def chunks(counts, limit=CHUNK):
    """Split runs into consecutive groups whose total length is about `limit`."""
    cum = np.cumsum(counts)
    start = 0
    while start < len(counts):
        base = cum[start - 1] if start else 0
        end = max(int(np.searchsorted(cum, base + limit, side='right')), start + 1)
        yield start, end
        start = end


def union_find(n, a, b):
    """Connected-component labels for n items joined by the pairs (a, b)."""
    labels = np.arange(n)
    while True:
        la, lb = labels[a], labels[b]
        diff = la != lb
        if not diff.any():
            return labels
        lo = np.minimum(la[diff], lb[diff])
        hi = np.maximum(la[diff], lb[diff])
        np.minimum.at(labels, hi, lo)
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped


def norm(v):
    return np.sqrt((v * v).sum(-1))


def segment_distance(p, a, b):
    """Distance from points p to segments a-b (all (n, 3))."""
    ab = b - a
    t = ((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30)
    t = np.clip(t, 0.0, 1.0)
    return norm(p - (a + t[:, None] * ab))


# Hard edges ----------------------------------------------------------------

def find_hard_edges(mesh, rule, angle_threshold, boundary_edges_are_hard):
    """Boolean per mesh edge: is it a hard edge under the chosen rule?"""
    edge_faces = np.bincount(mesh['loop_edge'], minlength=len(mesh['edges']))

    if rule == 'SHARP':
        hard = mesh['sharp'].copy()
    elif rule == 'ANGLE':
        normals = mesh['face_normals']
        loop_face = mesh['loop_face']
        reference = np.zeros(len(mesh['edges']), dtype=np.int64)
        reference[mesh['loop_edge']] = loop_face
        dots = (normals[loop_face] * normals[reference[mesh['loop_edge']]]).sum(1)
        min_dot = np.ones(len(mesh['edges']))
        np.minimum.at(min_dot, mesh['loop_edge'], dots)
        hard = min_dot < math.cos(math.radians(angle_threshold))
    else:
        raise EdgeDistanceError(f"Unknown HARD_EDGE_RULE {rule!r}, use 'SHARP' or 'ANGLE'")

    if boundary_edges_are_hard:
        hard |= edge_faces == 1
    return hard & (edge_faces > 0)


# Triangle sides ------------------------------------------------------------

def triangle_sides(mesh, uv_px, hard):
    """One record per triangle side, oriented so v0 < v1.

    Record r belongs to triangle r % T and is side r // T, which runs from
    corner k to corner k + 1.
    """
    tris = mesh['tri_loops']
    T = len(tris)
    loop_a = np.concatenate([tris[:, k] for k in range(3)])
    loop_b = np.concatenate([tris[:, (k + 1) % 3] for k in range(3)])
    va = mesh['loop_vert'][loop_a]
    vb = mesh['loop_vert'][loop_b]
    swap = va > vb
    loop_a, loop_b = np.where(swap, loop_b, loop_a), np.where(swap, loop_a, loop_b)
    va, vb = np.minimum(va, vb), np.maximum(va, vb)

    n_verts = len(mesh['positions'])
    key = va.astype(np.int64) * n_verts + vb
    edges = np.sort(mesh['edges'], axis=1).astype(np.int64)
    hard_keys = edges[hard, 0] * n_verts + edges[hard, 1]

    return {
        'tri': np.tile(np.arange(T), 3),
        'va': va, 'vb': vb,
        'uva': uv_px[loop_a], 'uvb': uv_px[loop_b],
        'key': key,
        'hard': np.isin(key, hard_keys),
    }


def uv_islands_and_seams(n_tris, sides):
    """Island label per triangle, and the pairs of sides that meet at a UV seam."""
    q = np.round(np.concatenate([sides['uva'], sides['uvb']], axis=1) * 256).astype(np.int64)
    order = np.lexsort((q[:, 3], q[:, 2], q[:, 1], q[:, 0], sides['key']))
    key, q = sides['key'][order], q[order]
    same_key = key[1:] == key[:-1]
    same_uv = (q[1:] == q[:-1]).all(1)

    join = same_key & same_uv
    tri = sides['tri']
    islands = union_find(n_tris, tri[order[:-1][join]], tri[order[1:][join]])

    seam = same_key & ~same_uv
    return islands, order[:-1][seam], order[1:][seam]


# Rasterization -------------------------------------------------------------

def rasterize(uv_tris, res):
    """Assign each texel the triangle (and barycentrics) whose UV area holds it.

    Texels just outside every triangle (within DILATE_RADIUS) take the nearest
    triangle with extrapolated barycentrics. Also returns how many texels are
    strictly inside more than one triangle (overlapping UVs).
    """
    N = res * res
    p0, p1, p2 = uv_tris[:, 0], uv_tris[:, 1], uv_tris[:, 2]
    v0, v1 = p1 - p0, p2 - p0
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    side_len = np.stack([norm(p2 - p1), norm(p2 - p0), norm(p1 - p0)], axis=1)
    height = np.abs(den)[:, None] / np.maximum(side_len, 1e-30)

    lo = uv_tris.min(1) - DILATE_RADIUS - 0.5
    hi = uv_tris.max(1) + DILATE_RADIUS - 0.5
    x0 = np.clip(np.ceil(lo[:, 0]), 0, res).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1]), 0, res).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0]), -1, res - 1).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1]), -1, res - 1).astype(np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    counts = np.where((np.abs(den) > 1e-12) & (w > 0) & (h > 0), w * h, 0)
    tri_ids = np.nonzero(counts)[0]
    counts = counts[tri_ids]

    best_dist = np.full(N, np.inf)
    best_tri = np.full(N, -1, dtype=np.int64)
    best_bary = np.zeros((N, 3))
    hits = np.zeros(N, dtype=np.int64)

    for s, e in chunks(counts):
        run, local = expand(counts[s:e])
        t = tri_ids[s:e][run]
        x = x0[t] + local % w[t]
        y = y0[t] + local // w[t]
        cx = x + 0.5 - p0[t, 0]
        cy = y + 0.5 - p0[t, 1]
        l1 = (cx * v1[t, 1] - v1[t, 0] * cy) / den[t]
        l2 = (v0[t, 0] * cy - cx * v0[t, 1]) / den[t]
        bary = np.stack([1.0 - l1 - l2, l1, l2], axis=1)
        signed = (bary * height[t]).min(1)

        keep = signed >= -DILATE_RADIUS
        t, bary, signed = t[keep], bary[keep], signed[keep]
        tex = (y * res + x)[keep]
        hits += np.bincount(tex[signed > 1e-3], minlength=N)

        dist = np.maximum(-signed, 0.0)
        order = np.lexsort((dist, tex))
        first = np.ones(order.size, dtype=bool)
        first[1:] = tex[order][1:] != tex[order][:-1]
        pick = order[first]
        better = dist[pick] <= best_dist[tex[pick]]
        pick = pick[better]
        best_dist[tex[pick]] = dist[pick]
        best_tri[tex[pick]] = t[pick]
        best_bary[tex[pick]] = bary[pick]

    return best_tri, best_bary, int((hits > 1).sum())


# Texel graph ---------------------------------------------------------------

def texel_samples(pa, pb, step=0.5):
    """Points every `step` pixels along UV segments: (segment index, t, point)."""
    lengths = norm(pb - pa)
    counts = np.ceil(lengths / step).astype(np.int64) + 1
    run, local = expand(counts)
    t = local / (counts[run] - 1)
    return run, t, pa[run] + t[:, None] * (pb[run] - pa[run])


def neighborhood(points, res):
    """For each point, the texels of its 3x3 neighborhood: (point index, texel)."""
    tx = np.floor(points[:, 0]).astype(np.int64)
    ty = np.floor(points[:, 1]).astype(np.int64)
    idx, tex = [], []
    for dy, dx in NEIGHBORHOOD:
        x, y = tx + dx, ty + dy
        ok = (x >= 0) & (x < res) & (y >= 0) & (y < res)
        idx.append(np.nonzero(ok)[0])
        tex.append(y[ok] * res + x[ok])
    return np.concatenate(idx), np.concatenate(tex)


def grid_graph(texels, res, pos, island, scale):
    """Edges between neighboring texels of the same island, plus triangles of
    those edges for the eikonal update."""
    N = res * res
    ty, tx = np.divmod(texels, res)
    src, dst, valid = [], [], {}
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ny, nx = ty + dy, tx + dx
        ok = (ny < res) & (nx >= 0) & (nx < res)
        a = texels[ok]
        b = ny[ok] * res + nx[ok]
        same = island[b] == island[a]
        a, b = a[same], b[same]
        expected = math.hypot(dy, dx) * np.maximum(scale[a], scale[b])
        short = norm(pos[a] - pos[b]) <= EDGE_SLACK * expected + 1e-12
        a, b = a[short], b[short]
        src.append(a)
        dst.append(b)
        flag = np.zeros(N, dtype=bool)
        flag[a] = True
        valid[dy, dx] = flag

    ty, tx = np.divmod(texels, res)
    a = texels[(tx < res - 1) & (ty < res - 1)]
    b, c, d = a + 1, a + res, a + res + 1
    ab, ac, ad = valid[0, 1][a], valid[1, 0][a], valid[1, 1][a]
    bd, cd, bc = valid[1, 0][b], valid[0, 1][c], valid[1, -1][b]
    t1 = ab & bd & ad
    t2 = ad & cd & ac
    both = t1 & t2
    t3 = ab & bc & ac & ~both
    t4 = bd & cd & bc & ~both
    tris = np.concatenate([
        np.stack([a[t1], b[t1], d[t1]], 1),
        np.stack([a[t2], d[t2], c[t2]], 1),
        np.stack([a[t3], b[t3], c[t3]], 1),
        np.stack([b[t4], d[t4], c[t4]], 1),
    ])
    return np.concatenate(src), np.concatenate(dst), tris


def seam_bridges(sides, seam_a, seam_b, tri_island, res, pos, island, scale):
    """Edges between texels on either side of each UV seam, so surface
    distance flows across seams."""
    if len(seam_a) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    lengths = np.maximum(norm(sides['uvb'][seam_a] - sides['uva'][seam_a]),
                         norm(sides['uvb'][seam_b] - sides['uva'][seam_b]))
    counts = np.ceil(lengths / 0.5).astype(np.int64) + 1

    src, dst = [], []
    for s, e in chunks(counts, CHUNK // 16):
        run, local = expand(counts[s:e])
        t = (local / (counts[s:e][run] - 1))[:, None]
        sa, sb = seam_a[s:e][run], seam_b[s:e][run]
        pt_a = sides['uva'][sa] + t * (sides['uvb'][sa] - sides['uva'][sa])
        pt_b = sides['uva'][sb] + t * (sides['uvb'][sb] - sides['uva'][sb])
        isl_a = tri_island[sides['tri'][sa]]
        isl_b = tri_island[sides['tri'][sb]]
        # Each side's center texel connects to the other side's 3x3 neighborhood.
        for here, there, isl_here, isl_there in ((pt_a, pt_b, isl_a, isl_b),
                                                 (pt_b, pt_a, isl_b, isl_a)):
            hx = np.clip(np.floor(here[:, 0]).astype(np.int64), 0, res - 1)
            hy = np.clip(np.floor(here[:, 1]).astype(np.int64), 0, res - 1)
            center = hy * res + hx
            idx, tex = neighborhood(there, res)
            c = center[idx]
            ok = (island[c] == isl_here[idx]) & (island[tex] == isl_there[idx])
            c, tex = c[ok], tex[ok]
            ok = norm(pos[c] - pos[tex]) <= EDGE_SLACK * np.maximum(scale[c], scale[tex])
            src.append(c[ok])
            dst.append(tex[ok])

    pairs = np.unique(np.stack([np.concatenate(src), np.concatenate(dst)], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


# Distance ------------------------------------------------------------------

def seed_distances(mesh, sides, tri_island, best_tri, texels, res, pos, island, scale):
    """Initial distances: exact in-plane distance for texels in triangles that
    touch a hard edge, plus texels near every hard edge in UV space."""
    N = res * res
    d = np.full(N, np.inf)
    world = mesh['world_positions']
    tri_verts = mesh['loop_vert'][mesh['tri_loops']]
    T = len(tri_verts)
    side_hard = sides['hard'].reshape(3, T).T

    t = best_tri[texels]
    for k in range(3):
        m = side_hard[t, k]
        tex, tk = texels[m], t[m]
        a = world[tri_verts[tk, k]]
        b = world[tri_verts[tk, (k + 1) % 3]]
        d[tex] = np.minimum(d[tex], segment_distance(pos[tex], a, b))

    hard = np.nonzero(sides['hard'])[0]
    pa, pb = sides['uva'][hard], sides['uvb'][hard]
    counts = np.ceil(norm(pb - pa) / 0.5).astype(np.int64) + 1
    for s, e in chunks(counts, CHUNK // 16):
        run, t_seg, pts = texel_samples(pa[s:e], pb[s:e])
        rec = hard[s:e][run]
        idx, tex = neighborhood(pts, res)
        rec = rec[idx]
        ok = island[tex] == tri_island[sides['tri'][rec]]
        tex, rec = tex[ok], rec[ok]
        dist = segment_distance(pos[tex], world[sides['va'][rec]], world[sides['vb'][rec]])
        near = dist <= 3.0 * scale[tex]
        np.minimum.at(d, tex[near], dist[near])
    return d


def eikonal(dA, dB, A, B, C):
    """Distance at C from a planar front through A and B (inf if not causal)."""
    out = np.full(len(C), np.inf)
    e1, e2 = A - C, B - C
    g11, g12, g22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    det = g11 * g22 - g12 * g12
    ok = np.isfinite(dA) & np.isfinite(dB) & (det > 1e-9 * g11 * g22)
    if not ok.any():
        return out
    g11, g12, g22, det = g11[ok], g12[ok], g22[ok], det[ok]
    base = np.minimum(dA[ok], dB[ok])
    da, db = dA[ok] - base, dB[ok] - base
    q11, q12, q22 = g22 / det, -g12 / det, g11 / det
    s1, s2 = q11 + q12, q12 + q22
    a = s1 + s2
    b = s1 * da + s2 * db
    c = q11 * da * da + 2.0 * q12 * da * db + q22 * db * db - 1.0
    disc = b * b - a * c
    T = (b + np.sqrt(np.maximum(disc, 0.0))) / a
    # The front must arrive from inside the triangle: -Q(d - T) >= 0.
    w1 = -(q11 * (da - T) + q12 * (db - T))
    w2 = -(q12 * (da - T) + q22 * (db - T))
    tol = 1e-9 * (np.abs(w1) + np.abs(w2))
    causal = (disc >= 0) & (w1 >= -tol) & (w2 >= -tol)
    out[np.nonzero(ok)[0][causal]] = (T + base)[causal]
    return out


def propagate(d, n, src, dst, tris, pos, tol, band):
    """Lower d until it is stable: edge (Dijkstra-like) relaxations plus
    triangle eikonal updates.

    Texels are expanded roughly in order of distance, one band of width `band`
    at a time, so each texel is expanded about once instead of being lowered
    over and over as better estimates arrive from behind.
    """
    order = np.argsort(src, kind='stable')
    src, dst = src[order], dst[order]
    weight = norm(pos[src] - pos[dst])
    e_off = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])

    flat = tris.ravel()
    order = np.argsort(flat, kind='stable')
    tri_of = np.repeat(np.arange(len(tris)), 3)[order]
    t_off = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=n))])

    # Pending texels are split into `near` (below `horizon`, scanned every
    # band) and `far` (rescanned only when near runs out). state: 0 = not
    # pending, 1 = near, 2 = far. Far may hold stale or duplicate entries; stale
    # ones are dropped on refill and duplicates only cost repeated work.
    NONE, NEAR, FAR = 0, 1, 2
    state = np.zeros(n, dtype=np.int8)
    far = np.nonzero(np.isfinite(d))[0]
    state[far] = FAR
    near = far[:0]
    horizon = threshold = -np.inf
    iterations = 0
    while near.size or far.size:
        if not near.size:
            far = far[state[far] == FAR]
            if not far.size:
                break
            horizon = float(d[far].min()) + 8 * band
            m = d[far] < horizon
            near, far = far[m], far[~m]
            state[near] = NEAR
        due = d[near] <= threshold
        if not due.any():
            threshold = max(threshold + band, float(d[near].min()))
            continue
        iterations += 1
        changed, near = near[due], near[~due]
        state[changed] = NONE

        ei = ranges(e_off[changed], e_off[changed + 1] - e_off[changed])
        targets = [dst[ei]]
        values = [d[src[ei]] + weight[ei]]

        ti = np.unique(tri_of[ranges(t_off[changed], t_off[changed + 1] - t_off[changed])])
        tri = tris[ti]
        for k in range(3):
            a, b, c = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
            targets.append(c)
            values.append(eikonal(d[a], d[b], pos[a], pos[b], pos[c]))

        targets = np.concatenate(targets)
        values = np.concatenate(values)
        better = values < d[targets] - tol
        targets, values = targets[better], values[better]
        np.minimum.at(d, targets, values)
        improved = np.unique(targets)
        close = d[improved] < horizon
        to_near = improved[close & (state[improved] != NEAR)]
        to_far = improved[~close & (state[improved] == NONE)]
        state[to_near] = NEAR
        state[to_far] = FAR
        near = np.concatenate([near, to_near])
        far = np.concatenate([far, to_far])
    return iterations


def unreachable_parts(mesh, hard):
    """Number of connected mesh parts that contain no hard edge."""
    edges = mesh['edges']
    labels = union_find(len(mesh['positions']), edges[:, 0], edges[:, 1])
    used = np.unique(labels[mesh['loop_vert']])
    reached = np.unique(labels[edges[hard].ravel()])
    return int(np.isin(used, reached, invert=True).sum())


def dilate(values, filled, res, margin):
    """Extend filled texels outward by `margin` pixels with neighbor averages."""
    img = values.reshape(res, res).copy()
    filled = filled.reshape(res, res).copy()
    for _ in range(margin):
        total = np.zeros((res + 2, res + 2))
        count = np.zeros((res + 2, res + 2))
        for dy, dx in NEIGHBORHOOD:
            if dy == dx == 0:
                continue
            total[1 + dy:res + 1 + dy, 1 + dx:res + 1 + dx] += np.where(filled, img, 0.0)
            count[1 + dy:res + 1 + dy, 1 + dx:res + 1 + dx] += filled
        total, count = total[1:-1, 1:-1], count[1:-1, 1:-1]
        grow = ~filled & (count > 0)
        if not grow.any():
            break
        img[grow] = total[grow] / count[grow]
        filled |= grow
    return img.ravel()


def bake(mesh, res=RESOLUTION, rule=HARD_EDGE_RULE, angle_threshold=ANGLE_THRESHOLD,
         boundary_edges_are_hard=BOUNDARY_EDGES_ARE_HARD, margin=MARGIN, log=print):
    """Compute the edge distance map. Returns (values (res*res,), warnings)."""
    clock = time.perf_counter()
    warnings = []

    hard = find_hard_edges(mesh, rule, angle_threshold, boundary_edges_are_hard)
    if not hard.any():
        hint = ("Mark edges Sharp, or switch HARD_EDGE_RULE to 'ANGLE'. A Bevel modifier "
                "removes sharp edges unless it marks its own (Shading > Harden Normals / Mark Sharp).")
        raise EdgeDistanceError(f"No hard edges found with rule {rule!r}. {hint}")

    uv_px = mesh['loop_uv'] * res
    sides = triangle_sides(mesh, uv_px, hard)
    T = len(mesh['tri_loops'])
    tri_island, seam_a, seam_b = uv_islands_and_seams(T, sides)

    uv_tris = uv_px[mesh['tri_loops']]
    world_tris = mesh['world_positions'][mesh['loop_vert'][mesh['tri_loops']]]
    best_tri, best_bary, overlaps = rasterize(uv_tris, res)
    if overlaps:
        warnings.append(f"Overlapping UVs: {overlaps} texels are covered by more than "
                        "one face. The result there will be wrong.")
    log(f"  rasterized {T} triangles ({time.perf_counter() - clock:.1f}s)")

    texels = np.nonzero(best_tri >= 0)[0]
    if texels.size == 0:
        raise EdgeDistanceError("The UV map covers no texels inside the 0-1 UV square.")
    N = res * res
    t = best_tri[texels]
    pos = np.zeros((N, 3))
    pos[texels] = np.einsum('ni,nij->nj', best_bary[texels], world_tris[t])
    island = np.full(N, -1, dtype=np.int64)
    island[texels] = tri_island[t]

    area_3d = 0.5 * norm(np.cross(world_tris[:, 1] - world_tris[:, 0],
                                  world_tris[:, 2] - world_tris[:, 0]))
    e1, e2 = uv_tris[:, 1] - uv_tris[:, 0], uv_tris[:, 2] - uv_tris[:, 0]
    area_uv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1])
    tri_scale = np.sqrt(area_3d / np.maximum(area_uv, 1e-30))
    scale = np.zeros(N)
    scale[texels] = tri_scale[t]

    src, dst, grid_tris = grid_graph(texels, res, pos, island, scale)
    bridge_src, bridge_dst = seam_bridges(sides, seam_a, seam_b, tri_island, res, pos, island, scale)
    src, dst = (np.concatenate([src, dst, bridge_src, bridge_dst]),
                np.concatenate([dst, src, bridge_dst, bridge_src]))
    log(f"  built texel graph: {texels.size} texels, {src.size} edges, "
        f"{len(grid_tris)} triangles ({time.perf_counter() - clock:.1f}s)")

    d = seed_distances(mesh, sides, tri_island, best_tri, texels, res, pos, island, scale)
    texel_size = float(np.median(scale[texels]))
    iterations = propagate(d, N, src, dst, grid_tris, pos, tol=1e-7 * texel_size, band=texel_size)
    log(f"  propagated distance in {iterations} iterations ({time.perf_counter() - clock:.1f}s)")

    reached = np.isfinite(d[texels])
    if not reached.any():
        raise EdgeDistanceError("No texel could reach a hard edge. Is the UV map valid?")
    peak = float(d[texels][reached].max())
    values = np.zeros(N)
    values[texels] = np.where(reached, d[texels] / peak if peak > 0 else 0.0, 1.0)

    parts = unreachable_parts(mesh, hard)
    if parts:
        warnings.append(f"{parts} disconnected part(s) have no hard edges and are filled with 1.")

    filled = np.zeros(N, dtype=bool)
    filled[texels] = True
    values = dilate(values, filled, res, margin)
    log(f"  peak distance {peak:.4g} ({time.perf_counter() - clock:.1f}s)")
    return values, warnings


# Blender -------------------------------------------------------------------

def read_mesh(obj):
    """Evaluated mesh of obj as NumPy arrays, positions in world space."""
    if obj is None or obj.type != 'MESH':
        raise EdgeDistanceError("The active object must be a mesh.")
    uv_layer = obj.data.uv_layers.active
    if uv_layer is None:
        raise EdgeDistanceError(f"{obj.name!r} has no UV map.")
    uv_name = uv_layer.name

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    me = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        def get(collection, prop, n, width, dtype):
            arr = np.empty(n * width, dtype=dtype)
            collection.foreach_get(prop, arr)
            return arr.reshape(n, width) if width > 1 else arr

        if hasattr(me, 'calc_loop_triangles'):
            me.calc_loop_triangles()
        nv, ne, nl, nf, nt = (len(me.vertices), len(me.edges), len(me.loops),
                              len(me.polygons), len(me.loop_triangles))
        if nt == 0:
            raise EdgeDistanceError(f"{obj.name!r} has no faces.")

        uv_attr = me.attributes.get(uv_name)
        if uv_attr is None or uv_attr.domain != 'CORNER':
            raise EdgeDistanceError(f"UV map {uv_name!r} is missing after modifiers.")

        sharp = np.zeros(ne, dtype=bool)
        sharp_attr = me.attributes.get('sharp_edge')
        if sharp_attr is not None and sharp_attr.domain == 'EDGE' and sharp_attr.data_type == 'BOOLEAN':
            sharp_attr.data.foreach_get('value', sharp)

        positions = get(me.vertices, 'co', nv, 3, np.float32).astype(np.float64)
        mesh = {
            'positions': positions,
            'edges': get(me.edges, 'vertices', ne, 2, np.int32).astype(np.int64),
            'loop_vert': get(me.loops, 'vertex_index', nl, 1, np.int32).astype(np.int64),
            'loop_edge': get(me.loops, 'edge_index', nl, 1, np.int32).astype(np.int64),
            'loop_uv': get(uv_attr.data, 'vector', nl, 2, np.float32).astype(np.float64),
            'tri_loops': get(me.loop_triangles, 'loops', nt, 3, np.int32).astype(np.int64),
            'tri_face': get(me.loop_triangles, 'polygon_index', nt, 1, np.int32).astype(np.int64),
            'sharp': sharp,
        }
        loop_total = get(me.polygons, 'loop_total', nf, 1, np.int32)
    finally:
        obj_eval.to_mesh_clear()

    matrix = np.array(obj.matrix_world, dtype=np.float64)
    mesh['world_positions'] = positions @ matrix[:3, :3].T + matrix[:3, 3]
    mesh['loop_face'] = np.repeat(np.arange(nf), loop_total)
    mesh['face_normals'] = face_normals(mesh, nf)
    return mesh


def face_normals(mesh, n_faces):
    """World-space face normals, area-weighted from the triangles."""
    w = mesh['world_positions'][mesh['loop_vert'][mesh['tri_loops']]]
    tri_n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
    normals = np.zeros((n_faces, 3))
    np.add.at(normals, mesh['tri_face'], tri_n)
    return normals / np.maximum(norm(normals), 1e-30)[:, None]


def write_image(name, values, res):
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.new(name, res, res, alpha=False, float_buffer=True, is_data=True)
    else:
        image.source = 'GENERATED'
        image.generated_width = res
        image.generated_height = res
        image.use_generated_float = True
    image.colorspace_settings.is_data = True
    rgba = np.ones((res * res, 4), dtype=np.float32)
    rgba[:, :3] = values[:, None]
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    return image


def popup(title, lines, icon):
    def draw(self, context):
        for line in lines:
            self.layout.label(text=line)
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def main():
    obj = bpy.context.view_layer.objects.active
    start = time.perf_counter()
    try:
        print(f"Edge Distance: baking {obj.name if obj else None!r} at {RESOLUTION}px")
        mesh = read_mesh(obj)
        values, warnings = bake(mesh)
        image = write_image(f"{obj.name}_EdgeDistance", values, RESOLUTION)
    except EdgeDistanceError as error:
        print(f"Edge Distance: {error}")
        popup("Edge Distance failed", [str(error)], 'ERROR')
        return
    for warning in warnings:
        print(f"Edge Distance warning: {warning}")
    done = f"Wrote {image.name!r} in {time.perf_counter() - start:.1f}s (not saved to disk yet)."
    print(f"Edge Distance: {done}")
    popup("Edge Distance", [done] + warnings, 'ERROR' if warnings else 'INFO')


if __name__ == '__main__' and bpy is not None:
    main()
