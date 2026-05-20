/*
 * tools/klotski_a_star.c — Experiment E15 deterministic oracle for the
 * simplified 4x5 Klotski puzzle from demos/character-level/klotski.
 *
 * The simplified Klotski state space is small enough for plain BFS
 * (every state is enumerable; we use a closed-set hash table) so we
 * don't actually need A*.  The filename uses "_a_star" to match the
 * E15 pre-registration text (`tools/klotski_a_star.c`) but the engine
 * is breadth-first search, which guarantees optimal solutions by
 * construction.
 *
 * Board encoding (20 chars + null):
 *   '.' = empty
 *   'A' = the 2x2 target block (must reach bottom-centre)
 *   'B', 'C' = 1x2 vertical blocks
 *   'D', 'E' = 1x1 cells in row 0
 *   'F', 'G' = 1x1 cells in row 1
 *
 * Goal: the 'A' block occupies rows ROWS-2..ROWS-1, cols 1..2.
 *
 * Move encoding: "<block_id><direction>" two-char tokens, e.g. "AD" =
 * move block A down, comma-separated for the full solution string,
 * e.g. "AD,FU,GL".
 *
 * Pure C99, no deps beyond libc/libm.
 *
 * Usage:
 *   klotski_a_star --count N [--seed S] [--difficulty mixed|easy|medium|hard]
 *
 * Emits N JSON lines to stdout, one per solved position, of the form:
 *
 *   {"state": "<20-char board>", "solution": "<move,move,...>",
 *    "moves": <int>, "scramble_depth": <int>}
 *
 * Difficulty buckets (by BFS-optimal solution length):
 *   easy   = moves in [1, 6]
 *   medium = moves in [7, 14]
 *   hard   = moves in [15, 30]
 *   mixed  = 30% easy / 50% medium / 20% hard
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>

#define ROWS 4
#define COLS 5
#define BOARD_SIZE (ROWS * COLS)
#define EMPTY '.'
#define MAX_MOVE_TOK 4   /* "AD," etc. */
#define MAX_SOLUTION 64  /* maximum optimal solution length we'll search */

typedef char Board[BOARD_SIZE];

static int cell(int r, int c) { return r * COLS + c; }

static void board_clone(Board dst, const Board src) {
    memcpy(dst, src, BOARD_SIZE);
}

static void dir_delta(char d, int *dr, int *dc) {
    *dr = 0; *dc = 0;
    if (d == 'U') *dr = -1;
    else if (d == 'D') *dr = 1;
    else if (d == 'L') *dc = -1;
    else if (d == 'R') *dc = 1;
}

static int can_move(const Board b, char block_id, char direction) {
    int dr, dc;
    dir_delta(direction, &dr, &dc);
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) {
            if (b[cell(r, c)] != block_id) continue;
            int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return 0;
            char dst = b[cell(nr, nc)];
            if (dst != EMPTY && dst != block_id) return 0;
        }
    return 1;
}

static void apply_move(Board b, char block_id, char direction) {
    int dr, dc;
    dir_delta(direction, &dr, &dc);
    int positions[BOARD_SIZE][2];
    int n = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            if (b[cell(r, c)] == block_id) {
                positions[n][0] = r;
                positions[n][1] = c;
                n++;
            }
    for (int i = 0; i < n; i++) b[cell(positions[i][0], positions[i][1])] = EMPTY;
    for (int i = 0; i < n; i++)
        b[cell(positions[i][0] + dr, positions[i][1] + dc)] = block_id;
}

static int is_goal(const Board b) {
    return b[cell(ROWS - 2, 1)] == 'A' && b[cell(ROWS - 2, 2)] == 'A' &&
           b[cell(ROWS - 1, 1)] == 'A' && b[cell(ROWS - 1, 2)] == 'A';
}

static void init_canonical(Board b) {
    /* Same starting/goal position as the demo. */
    memset(b, EMPTY, BOARD_SIZE);
    b[cell(ROWS - 2, 1)] = 'A';
    b[cell(ROWS - 2, 2)] = 'A';
    b[cell(ROWS - 1, 1)] = 'A';
    b[cell(ROWS - 1, 2)] = 'A';
    b[cell(0, 0)] = 'B';
    b[cell(1, 0)] = 'B';
    b[cell(0, 3)] = 'C';
    b[cell(1, 3)] = 'C';
    b[cell(0, 1)] = 'D';
    b[cell(0, 2)] = 'E';
    b[cell(1, 1)] = 'F';
    b[cell(1, 2)] = 'G';
}

/* ─── BFS with hash-table closed set ──────────────────────────────────
 *
 * State count for the simplified Klotski is on the order of a few
 * hundred thousand reachable positions.  An open-addressed hash table
 * sized for ~2M entries comfortably accommodates that with low LF.
 *
 * State key: the 20-char board string itself (interned via the
 * closed set as a flat memcpy).  We allocate 21-byte slots (board +
 * nul) keyed by FNV-1a 64-bit of the 20 chars; chain via parent index
 * + last-move token stored in parallel arrays.  This is enough to
 * reconstruct the move sequence by walking the parent chain.
 */

typedef struct {
    char    board[BOARD_SIZE];
    int     parent;     /* -1 for the root */
    int     depth;
    char    move[3];    /* "AD\0" — block + direction */
} BfsNode;

typedef struct {
    BfsNode *nodes;
    int      n_nodes;
    int      capacity;
    /* Open-addressed hash table mapping board -> node index. */
    int     *hash;       /* -1 = empty */
    int      hash_mask;  /* capacity - 1 (power of two) */
} BfsCtx;

static uint64_t fnv64(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int bfs_lookup(const BfsCtx *ctx, const char *board) {
    uint64_t h = fnv64(board, BOARD_SIZE);
    int mask = ctx->hash_mask;
    int idx = (int)(h & (uint64_t)mask);
    while (ctx->hash[idx] != -1) {
        if (memcmp(ctx->nodes[ctx->hash[idx]].board, board, BOARD_SIZE) == 0)
            return ctx->hash[idx];
        idx = (idx + 1) & mask;
    }
    return -1;
}

static int bfs_insert(BfsCtx *ctx, const char *board, int parent,
                      int depth, const char *move) {
    if (ctx->n_nodes >= ctx->capacity) return -1;  /* full */
    int nid = ctx->n_nodes++;
    BfsNode *n = &ctx->nodes[nid];
    memcpy(n->board, board, BOARD_SIZE);
    n->parent = parent;
    n->depth  = depth;
    n->move[0] = move ? move[0] : 0;
    n->move[1] = move ? move[1] : 0;
    n->move[2] = 0;
    /* Insert into the hash table. */
    uint64_t h = fnv64(board, BOARD_SIZE);
    int mask = ctx->hash_mask;
    int idx = (int)(h & (uint64_t)mask);
    while (ctx->hash[idx] != -1) idx = (idx + 1) & mask;
    ctx->hash[idx] = nid;
    return nid;
}

/* Search for the unique block ids present on the board. */
static int collect_blocks(const Board b, char *blocks) {
    char seen[256] = {0};
    int n = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        unsigned char ch = (unsigned char)b[i];
        if (ch != EMPTY && !seen[ch]) {
            seen[ch] = 1;
            blocks[n++] = (char)ch;
        }
    }
    return n;
}

/* Reconstruct the comma-joined move sequence by walking parents.
 * `out` must be >= MAX_MOVE_TOK * MAX_SOLUTION + 1 bytes. */
static void reconstruct_path(const BfsCtx *ctx, int goal_id, char *out) {
    /* Walk to root, collecting move tokens in reverse. */
    char buf[MAX_SOLUTION][3];
    int n = 0;
    int cur = goal_id;
    while (cur >= 0 && n < MAX_SOLUTION) {
        const BfsNode *nd = &ctx->nodes[cur];
        if (nd->parent < 0) break;  /* root has no move */
        buf[n][0] = nd->move[0];
        buf[n][1] = nd->move[1];
        buf[n][2] = 0;
        n++;
        cur = nd->parent;
    }
    /* Reverse and join with commas. */
    size_t pos = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (pos > 0) out[pos++] = ',';
        out[pos++] = buf[i][0];
        out[pos++] = buf[i][1];
    }
    out[pos] = '\0';
}

/* BFS until goal is reached or queue exhausted.  Returns the
 * solution length, or -1 if unsolvable / capacity exceeded.
 * `out` (if non-NULL) is written with the move sequence. */
static int solve_bfs(const Board start, char *out, int *out_visited) {
    static const int CAPACITY = 1 << 20;          /* 1M nodes */
    static const int HASH_CAP = 1 << 21;          /* 2M slots, LF<=0.5 */
    BfsCtx ctx;
    ctx.nodes = (BfsNode *)malloc((size_t)CAPACITY * sizeof(BfsNode));
    ctx.hash  = (int *)malloc((size_t)HASH_CAP * sizeof(int));
    if (!ctx.nodes || !ctx.hash) {
        free(ctx.nodes); free(ctx.hash);
        return -1;
    }
    ctx.n_nodes = 0;
    ctx.capacity = CAPACITY;
    ctx.hash_mask = HASH_CAP - 1;
    for (int i = 0; i < HASH_CAP; i++) ctx.hash[i] = -1;

    /* Insert root. */
    if (bfs_insert(&ctx, start, -1, 0, NULL) < 0) {
        free(ctx.nodes); free(ctx.hash);
        return -1;
    }

    int head = 0;
    int found = -1;
    while (head < ctx.n_nodes) {
        const BfsNode *cur = &ctx.nodes[head++];
        if (is_goal(cur->board)) { found = head - 1; break; }
        if (cur->depth >= MAX_SOLUTION) continue;

        Board cur_board;
        board_clone(cur_board, cur->board);
        char blocks[16];
        int nb = collect_blocks(cur_board, blocks);
        for (int bi = 0; bi < nb; bi++) {
            char bid = blocks[bi];
            const char dirs[] = "UDLR";
            for (int di = 0; di < 4; di++) {
                char d = dirs[di];
                if (!can_move(cur_board, bid, d)) continue;
                Board next;
                board_clone(next, cur_board);
                apply_move(next, bid, d);
                if (bfs_lookup(&ctx, next) >= 0) continue;
                char mv[2] = { bid, d };
                if (bfs_insert(&ctx, next, head - 1, cur->depth + 1, mv) < 0) {
                    /* Capacity exhausted. */
                    if (out_visited) *out_visited = ctx.n_nodes;
                    free(ctx.nodes); free(ctx.hash);
                    return -1;
                }
            }
        }
    }
    if (out_visited) *out_visited = ctx.n_nodes;
    int sol_len = -1;
    if (found >= 0) {
        sol_len = ctx.nodes[found].depth;
        if (out) reconstruct_path(&ctx, found, out);
    } else if (out) {
        out[0] = '\0';
    }
    free(ctx.nodes); free(ctx.hash);
    return sol_len;
}

/* ─── Random board generation by scramble walk from goal ────────────── */

static uint64_t lehmer_state;
static uint32_t lehmer_next(void) {
    lehmer_state ^= lehmer_state >> 12;
    lehmer_state ^= lehmer_state << 25;
    lehmer_state ^= lehmer_state >> 27;
    return (uint32_t)(lehmer_state * 0x2545F4914F6CDD1DULL >> 32);
}
static void lehmer_seed(uint64_t s) {
    lehmer_state = s ? s : 0x9E3779B97F4A7C15ULL;
    (void)lehmer_next();
    (void)lehmer_next();
}

static void scramble(Board b, int depth) {
    /* Random valid moves; do not undo the last move directly. */
    char last_block = 0, last_dir = 0;
    int done = 0, safety = depth * 16;
    while (done < depth && safety-- > 0) {
        char blocks[16];
        int nb = collect_blocks(b, blocks);
        char bid = blocks[lehmer_next() % nb];
        const char dirs[] = "UDLR";
        char d = dirs[lehmer_next() % 4];
        /* Reject immediate inverse to keep depth ~= move count. */
        char inv = 0;
        if (last_dir == 'U') inv = 'D';
        else if (last_dir == 'D') inv = 'U';
        else if (last_dir == 'L') inv = 'R';
        else if (last_dir == 'R') inv = 'L';
        if (bid == last_block && d == inv) continue;
        if (!can_move(b, bid, d)) continue;
        apply_move(b, bid, d);
        last_block = bid;
        last_dir = d;
        done++;
    }
}

typedef enum { D_EASY = 0, D_MEDIUM, D_HARD, D_MIXED } DifficultyKind;
static int scramble_depth_for(DifficultyKind k) {
    switch (k) {
    case D_EASY:   return 4;
    case D_MEDIUM: return 10;
    case D_HARD:   return 22;
    default:       return 10;
    }
}
static int sol_lo(DifficultyKind k) {
    switch (k) {
    case D_EASY:   return 1;
    case D_MEDIUM: return 7;
    case D_HARD:   return 15;
    default:       return 1;
    }
}
static int sol_hi(DifficultyKind k) {
    switch (k) {
    case D_EASY:   return 6;
    case D_MEDIUM: return 14;
    case D_HARD:   return 30;
    default:       return 30;
    }
}

static DifficultyKind mixed_pick(uint32_t r) {
    uint32_t p = r % 100;
    if (p < 30) return D_EASY;
    if (p < 80) return D_MEDIUM;
    return D_HARD;
}

static void encode_state(const Board b, char *out) {
    memcpy(out, b, BOARD_SIZE);
    out[BOARD_SIZE] = '\0';
}

/* Escape the state for JSON; only `\` and `"` would need escaping but
 * our state alphabet (uppercase letters + '.') is JSON-safe directly. */

static void usage(FILE *f) {
    fprintf(f,
        "klotski_a_star -- Experiment E15 deterministic oracle (BFS-optimal)\n"
        "Usage: klotski_a_star [options]\n"
        "  --count N         Emit N solved positions (default 10)\n"
        "  --seed S          PRNG seed (default 1337)\n"
        "  --difficulty K    easy | medium | hard | mixed (default mixed)\n"
        "  --quiet           Suppress progress on stderr\n"
        "  --self-test       Solve 10 scrambled positions; exit 0 on success\n");
}

static int parse_difficulty(const char *s) {
    if (!s) return D_MIXED;
    if (!strcmp(s, "easy"))   return D_EASY;
    if (!strcmp(s, "medium")) return D_MEDIUM;
    if (!strcmp(s, "hard"))   return D_HARD;
    return D_MIXED;
}

int main(int argc, char **argv) {
    int count = 10;
    uint64_t seed = 1337;
    DifficultyKind diff = D_MIXED;
    int quiet = 0;
    int self_test = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--count") && i + 1 < argc) {
            count = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--difficulty") && i + 1 < argc) {
            diff = parse_difficulty(argv[++i]);
        } else if (!strcmp(argv[i], "--quiet")) {
            quiet = 1;
        } else if (!strcmp(argv[i], "--self-test")) {
            self_test = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "klotski_a_star: unknown arg '%s'\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }

    lehmer_seed(seed);

    if (self_test) {
        int ok = 0;
        for (int i = 0; i < 10; i++) {
            Board b;
            init_canonical(b);
            scramble(b, 6);
            char sol[MAX_MOVE_TOK * MAX_SOLUTION + 1];
            int visited = 0;
            int len = solve_bfs(b, sol, &visited);
            if (len < 0) {
                fprintf(stderr, "self-test: solve %d failed (visited %d)\n",
                        i, visited);
                return 1;
            }
            /* Verify the solution drives the board to goal. */
            Board check;
            board_clone(check, b);
            const char *p = sol;
            int verified = 0;
            while (*p) {
                while (*p == ',') p++;
                if (!*p) break;
                char bid = *p++;
                if (!*p) { verified = -1; break; }
                char dir = *p++;
                if (!can_move(check, bid, dir)) { verified = -1; break; }
                apply_move(check, bid, dir);
                verified++;
            }
            if (verified != len || !is_goal(check)) {
                fprintf(stderr,
                    "self-test: verify failed (len=%d, applied=%d, goal=%d)\n",
                    len, verified, is_goal(check));
                return 1;
            }
            ok++;
        }
        fprintf(stderr, "klotski_a_star self-test: %d/10 OK\n", ok);
        return ok == 10 ? 0 : 1;
    }

    long emitted = 0, attempted = 0;
    clock_t t0 = clock();
    while (emitted < count) {
        DifficultyKind d = (diff == D_MIXED) ? mixed_pick(lehmer_next()) : diff;
        Board b;
        init_canonical(b);
        int sd = scramble_depth_for(d);
        scramble(b, sd);
        attempted++;
        if (is_goal(b)) continue;  /* scramble was a no-op or returned to goal */
        char sol[MAX_MOVE_TOK * MAX_SOLUTION + 1];
        int visited = 0;
        int len = solve_bfs(b, sol, &visited);
        if (len < 0) {
            if (!quiet) fprintf(stderr,
                "[klotski] skip: solver failed (visited %d)\n", visited);
            continue;
        }
        /* Filter by the difficulty's true-optimal range. */
        if (diff != D_MIXED) {
            if (len < sol_lo(d) || len > sol_hi(d)) continue;
        }
        char st[BOARD_SIZE + 1];
        encode_state(b, st);
        printf("{\"state\":\"%s\",\"solution\":\"%s\",\"moves\":%d,"
               "\"scramble_depth\":%d}\n",
               st, sol, len, sd);
        emitted++;
        if (!quiet && (emitted % 100 == 0)) {
            double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
            fprintf(stderr,
                "[klotski] %ld / %d emitted (%.1fs, attempts %ld)\n",
                emitted, count, el, attempted);
        }
    }
    if (!quiet) {
        double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
        fprintf(stderr, "[klotski] DONE: %ld emitted, %ld attempts, %.2fs\n",
                emitted, attempted, el);
    }
    return 0;
}
