/*
 * microgpt_vr.c — C99 Vietoris-Rips Persistent Cohomology Engine
 *
 * Ported from EnX-cpp vr_engine.hpp (C++17, 695 lines).
 * Fixed at 12 dimensions, 64 max points.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 */

#include "microgpt_vr.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/* =========================================================================
 * Internal Types
 * ========================================================================= */

/* Simplex: vertex list + filtration value */
typedef struct {
    int   vertices[3];  /* Max 3 vertices (triangle) */
    int   n_verts;      /* 1=vertex, 2=edge, 3=triangle */
    float filtration;
} VRSimplex;

/* Sparse column over F₂ (sorted row indices) */
typedef struct {
    int entries[VR_MAX_SIMPLICES];
    int count;
} SparseColumn;

/* Internal engine state for a single computation */
typedef struct {
    /* Stage 1: Distance matrix (lower triangular, squared) */
    float dist_matrix[VR_MAX_PTS * (VR_MAX_PTS - 1) / 2];
    int   num_points;
    int   n_dims;

    /* Stage 2: Filtration */
    VRSimplex filtration[VR_MAX_SIMPLICES];
    int       filt_count;

    /* Stage 3: Pairing */
    int paired_birth[VR_MAX_SIMPLICES];  /* simplex → paired partner */
    int paired_death[VR_MAX_SIMPLICES];
} VRState;

/* =========================================================================
 * Helper Functions
 * ========================================================================= */

VRPoint vr_make_point(const float *coords, int n_dims, int id) {
    VRPoint p;
    memset(&p, 0, sizeof(p));
    if (n_dims > VR_MAX_DIMS) n_dims = VR_MAX_DIMS;
    p.n_dims = n_dims;
    p.id = id;
    for (int i = 0; i < n_dims; i++) p.coords[i] = coords[i];
    return p;
}

static size_t flat_idx(int i, int j) {
    return (size_t)i * (i - 1) / 2 + j;
}

static float get_dist(const VRState *st, int i, int j) {
    if (i == j) return 0.0f;
    if (i < j) { int tmp = i; i = j; j = tmp; }
    return st->dist_matrix[flat_idx(i, j)];
}

/* Simplex hash (golden ratio combiner) */
static size_t hash_verts(const int *verts, int n) {
    size_t h = 0;
    for (int i = 0; i < n; i++) {
        size_t vh = (size_t)(verts[i] * 2654435761u);
        h ^= vh + 0x9e3779b9u + (h << 6) + (h >> 2);
    }
    return h;
}

/* =========================================================================
 * Stage 1: Distance Matrix
 * ========================================================================= */

static void compute_distance_matrix(VRState *st, const VRPoint *points,
                                    int n, int n_dims) {
    st->num_points = n;
    st->n_dims = n_dims;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float sum = 0.0f;
            for (int d = 0; d < n_dims; d++) {
                float diff = points[i].coords[d] - points[j].coords[d];
                sum += diff * diff;
            }
            st->dist_matrix[flat_idx(i, j)] = sum;
        }
    }
}

/* =========================================================================
 * Stage 2: Flag Complex Filtration
 * ========================================================================= */

/* Edge entry for sorting */
typedef struct { int u, v; float weight; } EdgeEntry;
typedef struct { int u, v, w; float weight; } TriEntry;

static int cmp_edge(const void *a, const void *b) {
    const EdgeEntry *ea = (const EdgeEntry *)a;
    const EdgeEntry *eb = (const EdgeEntry *)b;
    if (ea->weight < eb->weight) return -1;
    if (ea->weight > eb->weight) return 1;
    if (ea->u != eb->u) return ea->u - eb->u;
    return ea->v - eb->v;
}

static int cmp_tri(const void *a, const void *b) {
    const TriEntry *ta = (const TriEntry *)a;
    const TriEntry *tb = (const TriEntry *)b;
    if (ta->weight < tb->weight) return -1;
    if (ta->weight > tb->weight) return 1;
    if (ta->u != tb->u) return ta->u - tb->u;
    if (ta->v != tb->v) return ta->v - tb->v;
    return ta->w - tb->w;
}

static void build_filtration(VRState *st, int n, float max_radius_sq,
                             int max_dim) {
    st->filt_count = 0;

    /* Dim 0: vertices */
    for (int i = 0; i < n && st->filt_count < VR_MAX_SIMPLICES; i++) {
        VRSimplex *s = &st->filtration[st->filt_count++];
        s->vertices[0] = i;
        s->n_verts = 1;
        s->filtration = 0.0f;
    }

    if (max_dim < 1) return;

    /* Dim 1: edges */
    EdgeEntry edges[VR_MAX_EDGES];
    int n_edges = 0;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float d2 = st->dist_matrix[flat_idx(i, j)];
            if (d2 <= max_radius_sq && n_edges < VR_MAX_EDGES) {
                edges[n_edges].u = i;
                edges[n_edges].v = j;
                edges[n_edges].weight = d2;
                n_edges++;
            }
        }
    }

    qsort(edges, n_edges, sizeof(EdgeEntry), cmp_edge);

    for (int e = 0; e < n_edges && st->filt_count < VR_MAX_SIMPLICES; e++) {
        VRSimplex *s = &st->filtration[st->filt_count++];
        int mn = (edges[e].u < edges[e].v) ? edges[e].u : edges[e].v;
        int mx = (edges[e].u > edges[e].v) ? edges[e].u : edges[e].v;
        s->vertices[0] = mn;
        s->vertices[1] = mx;
        s->n_verts = 2;
        s->filtration = edges[e].weight;
    }

    if (max_dim < 2) return;

    /* Dim 2: triangles via bitmask adjacency */
    uint64_t adj[VR_MAX_PTS];
    memset(adj, 0, sizeof(adj));

    for (int e = 0; e < n_edges; e++) {
        int u = edges[e].u, v = edges[e].v;
        adj[u] |= (1ULL << v);
        adj[v] |= (1ULL << u);
    }

    TriEntry triangles[VR_MAX_TRIANGLES];
    int n_tris = 0;

    for (int e = 0; e < n_edges && n_tris < VR_MAX_TRIANGLES; e++) {
        int u = edges[e].u, v = edges[e].v;
        uint64_t common = adj[u] & adj[v];

        while (common && n_tris < VR_MAX_TRIANGLES) {
            int w = 0;
            /* Find lowest set bit */
            uint64_t lsb = common & (~common + 1);
            /* Count trailing zeros to get bit position */
            {
                uint64_t tmp = lsb;
                w = 0;
                while (tmp > 1) { tmp >>= 1; w++; }
            }
            common &= ~lsb;

            /* Canonical ordering */
            int arr[3] = {u, v, w};
            /* Sort 3-element array */
            if (arr[0] > arr[1]) { int t = arr[0]; arr[0] = arr[1]; arr[1] = t; }
            if (arr[1] > arr[2]) { int t = arr[1]; arr[1] = arr[2]; arr[2] = t; }
            if (arr[0] > arr[1]) { int t = arr[0]; arr[0] = arr[1]; arr[1] = t; }

            if (arr[0] == arr[1] || arr[1] == arr[2]) continue;

            float d_uv = edges[e].weight;
            float d_uw = get_dist(st, u, w);
            float d_vw = get_dist(st, v, w);
            float birth = d_uv;
            if (d_uw > birth) birth = d_uw;
            if (d_vw > birth) birth = d_vw;

            triangles[n_tris].u = arr[0];
            triangles[n_tris].v = arr[1];
            triangles[n_tris].w = arr[2];
            triangles[n_tris].weight = birth;
            n_tris++;
        }
    }

    /* Deduplicate triangles */
    qsort(triangles, n_tris, sizeof(TriEntry), cmp_tri);
    int deduped = 0;
    for (int i = 0; i < n_tris; i++) {
        if (i > 0 && triangles[i].u == triangles[i-1].u &&
            triangles[i].v == triangles[i-1].v &&
            triangles[i].w == triangles[i-1].w) continue;
        triangles[deduped++] = triangles[i];
    }
    n_tris = deduped;

    /* Re-sort by weight */
    qsort(triangles, n_tris, sizeof(TriEntry), cmp_tri);

    for (int t = 0; t < n_tris && st->filt_count < VR_MAX_SIMPLICES; t++) {
        VRSimplex *s = &st->filtration[st->filt_count++];
        s->vertices[0] = triangles[t].u;
        s->vertices[1] = triangles[t].v;
        s->vertices[2] = triangles[t].w;
        s->n_verts = 3;
        s->filtration = triangles[t].weight;
    }
}

/* =========================================================================
 * Stage 3: F₂ Persistent Cohomology Reduction
 * ========================================================================= */

/* Hash table for simplex → index lookup */
#define HASH_SIZE 16384
#define HASH_MASK (HASH_SIZE - 1)

typedef struct { size_t key; int value; int occupied; } HashEntry;

static int hash_lookup(const HashEntry *table, size_t key) {
    int idx = (int)(key & HASH_MASK);
    for (int probe = 0; probe < 128; probe++) {
        int i = (idx + probe) & HASH_MASK;
        if (!table[i].occupied) return -1;
        if (table[i].key == key) return table[i].value;
    }
    return -1;
}

static void hash_insert(HashEntry *table, size_t key, int value) {
    int idx = (int)(key & HASH_MASK);
    for (int probe = 0; probe < 128; probe++) {
        int i = (idx + probe) & HASH_MASK;
        if (!table[i].occupied) {
            table[i].key = key;
            table[i].value = value;
            table[i].occupied = 1;
            return;
        }
    }
}

/* Compute boundary of a simplex as sorted row indices */
static void compute_boundary(const VRSimplex *sigma, const HashEntry *idx_table,
                             int *entries, int *count) {
    *count = 0;
    if (sigma->n_verts < 2) return;

    for (int i = 0; i < sigma->n_verts; i++) {
        int face[3];
        int fc = 0;
        for (int j = 0; j < sigma->n_verts; j++) {
            if (j != i) face[fc++] = sigma->vertices[j];
        }
        size_t h = hash_verts(face, fc);
        int idx = hash_lookup(idx_table, h);
        if (idx >= 0) entries[(*count)++] = idx;
    }

    /* Sort entries */
    for (int i = 0; i < *count - 1; i++)
        for (int j = i + 1; j < *count; j++)
            if (entries[i] > entries[j]) {
                int tmp = entries[i]; entries[i] = entries[j]; entries[j] = tmp;
            }
}

/* F₂ addition: symmetric difference of two sorted arrays */
static void sparse_add(int *a, int *na, const int *b, int nb) {
    int result[VR_MAX_SIMPLICES];
    int rc = 0, i = 0, j = 0;
    while (i < *na && j < nb) {
        if (a[i] < b[j]) { result[rc++] = a[i++]; }
        else if (a[i] > b[j]) { result[rc++] = b[j++]; }
        else { i++; j++; } /* XOR cancels */
    }
    while (i < *na) result[rc++] = a[i++];
    while (j < nb) result[rc++] = b[j++];
    memcpy(a, result, rc * sizeof(int));
    *na = rc;
}

static void reduce(VRState *st) {
    int m = st->filt_count;

    for (int i = 0; i < m; i++) {
        st->paired_birth[i] = -1;
        st->paired_death[i] = -1;
    }

    /* Build simplex → index hash table */
    HashEntry *idx_table = (HashEntry *)calloc(HASH_SIZE, sizeof(HashEntry));
    if (!idx_table) return;

    for (int i = 0; i < m; i++) {
        size_t h = hash_verts(st->filtration[i].vertices,
                              st->filtration[i].n_verts);
        hash_insert(idx_table, h, i);
    }

    /* Build boundary columns (dynamically allocated) */
    int (*columns)[VR_MAX_SIMPLICES] = NULL;
    int *col_counts = NULL;

    columns = (int (*)[VR_MAX_SIMPLICES])calloc(m, sizeof(int[VR_MAX_SIMPLICES]));
    col_counts = (int *)calloc(m, sizeof(int));
    if (!columns || !col_counts) {
        free(idx_table); free(columns); free(col_counts);
        return;
    }

    for (int i = 0; i < m; i++) {
        compute_boundary(&st->filtration[i], idx_table,
                         columns[i], &col_counts[i]);
    }

    int *cleared = (int *)calloc(m, sizeof(int));
    if (!cleared) {
        free(idx_table); free(columns); free(col_counts);
        return;
    }

    /* Apparent pairs */
    for (int j = 0; j < m; j++) {
        if (st->filtration[j].n_verts < 2) continue;
        if (col_counts[j] == 0) continue;
        if (col_counts[j] == 1) {
            int pivot = columns[j][0];
            st->paired_birth[j] = pivot;
            st->paired_death[pivot] = j;
            cleared[pivot] = 1;
            cleared[j] = 1;
        }
    }

    /* Full column reduction (per dimension) */
    for (int dim = 1; dim <= 2; dim++) {
        /* Pivot map: pivot → column index */
        HashEntry *pivot_map = (HashEntry *)calloc(HASH_SIZE, sizeof(HashEntry));
        if (!pivot_map) continue;

        /* Seed with apparent pairs */
        for (int j = 0; j < m; j++) {
            if ((st->filtration[j].n_verts - 1) != dim) continue;
            if (st->paired_birth[j] >= 0 && col_counts[j] > 0) {
                int piv = columns[j][col_counts[j] - 1];
                hash_insert(pivot_map, (size_t)piv, j);
            }
        }

        for (int j = 0; j < m; j++) {
            if ((st->filtration[j].n_verts - 1) != dim) continue;
            if (cleared[j]) continue;

            while (col_counts[j] > 0) {
                int piv = columns[j][col_counts[j] - 1];
                int other = hash_lookup(pivot_map, (size_t)piv);
                if (other < 0) break;
                sparse_add(columns[j], &col_counts[j],
                           columns[other], col_counts[other]);
            }

            if (col_counts[j] > 0) {
                int piv = columns[j][col_counts[j] - 1];
                hash_insert(pivot_map, (size_t)piv, j);
                st->paired_birth[j] = piv;
                st->paired_death[piv] = j;
                cleared[piv] = 1;
            }
        }

        free(pivot_map);
    }

    free(idx_table);
    free(columns);
    free(col_counts);
    free(cleared);
}

/* =========================================================================
 * Stage 4: Extract Persistence Diagram
 * ========================================================================= */

static VRDiagram extract_diagram(const VRState *st, float min_persistence) {
    VRDiagram diag;
    diag.count = 0;
    int m = st->filt_count;

    for (int i = 0; i < m && diag.count < VR_MAX_INTERVALS; i++) {
        int dim = st->filtration[i].n_verts - 1;
        float birth = st->filtration[i].filtration;

        if (st->paired_death[i] >= 0) {
            float death = st->filtration[st->paired_death[i]].filtration;
            float pers = death - birth;
            if (pers > min_persistence) {
                diag.intervals[diag.count].dimension = dim;
                diag.intervals[diag.count].birth = birth;
                diag.intervals[diag.count].death = death;
                diag.count++;
            }
        } else if (st->paired_birth[i] < 0) {
            /* Essential feature */
            diag.intervals[diag.count].dimension = dim;
            diag.intervals[diag.count].birth = birth;
            diag.intervals[diag.count].death = FLT_MAX;
            diag.count++;
        }
    }

    return diag;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

void vr_engine_init(VREngine *engine, float max_radius, int max_dim,
                    int n_dims) {
    engine->max_radius = max_radius;
    engine->max_radius_sq = max_radius * max_radius;
    engine->max_dim = (max_dim >= 0 && max_dim <= 2) ? max_dim : 2;
    engine->n_dims = (n_dims > 0 && n_dims <= VR_MAX_DIMS) ? n_dims : VR_MAX_DIMS;
}

VRDiagram vr_compute(VREngine *engine, const VRPoint *points, int n_points,
                     float min_persistence) {
    VRDiagram empty;
    memset(&empty, 0, sizeof(empty));
    if (n_points <= 0 || n_points > VR_MAX_PTS) return empty;

    VRState *st = (VRState *)calloc(1, sizeof(VRState));
    if (!st) return empty;

    compute_distance_matrix(st, points, n_points, engine->n_dims);
    build_filtration(st, n_points, engine->max_radius_sq, engine->max_dim);
    reduce(st);
    VRDiagram diag = extract_diagram(st, min_persistence);

    free(st);
    return diag;
}

void vr_betti_numbers(VREngine *engine, const VRPoint *points, int n_points,
                      float at_radius, float min_persistence,
                      int betti_out[3]) {
    VRDiagram diag = vr_compute(engine, points, n_points, min_persistence);
    float r = (at_radius < 0.0f) ? engine->max_radius_sq : at_radius;
    betti_out[0] = vr_betti_at(&diag, 0, r);
    betti_out[1] = vr_betti_at(&diag, 1, r);
    betti_out[2] = vr_betti_at(&diag, 2, r);
}

int vr_betti_at(const VRDiagram *diagram, int dim, float filtration) {
    int b = 0;
    for (int i = 0; i < diagram->count; i++) {
        if (diagram->intervals[i].dimension == dim &&
            diagram->intervals[i].birth <= filtration &&
            filtration < diagram->intervals[i].death)
            b++;
    }
    return b;
}
