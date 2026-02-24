/*
 * transfer_demo — Transfer Learning Between Organelles (K-12)
 *
 * Tests whether internal transformer representations trained on one game
 * transfer to another.  Three conditions are compared:
 *
 *   1. SCRATCH   — Train Othello Player from scratch (baseline)
 *   2. TRANSFER  — Train TicTacToe Player, transfer internal weights to
 *                  fresh Othello model (NO fine-tuning on Othello)
 *   3. RANDOM    — Untrained model (negative control)
 *
 * Both games share architecture 48/4/3/128/192 so transfer is dimension-safe.
 * The internal transformer layers (attention Q/K/V/O, MLP fc1/fc2, and wpe)
 * are vocab-agnostic.  Only wte and lm_head depend on vocabulary.
 *
 * This tests raw representational transfer — can a model that learned to
 * play TicTacToe perform better at Othello than a random model, even
 * without any Othello-specific fine-tuning?
 */

#include "microgpt.h"
#include "microgpt_organelle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Othello 6×6 game logic ---- */

#define BSIZ 6
#define CELLS (BSIZ * BSIZ)
#define EMPTY '.'
#define BLACK 'B'
#define WHITE 'W'
#define TEMP 0.3
#define GEN_LEN 12

static const int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                               {0, 1},   {1, -1}, {1, 0},  {1, 1}};

static void board_init(char *b) {
  memset(b, EMPTY, CELLS);
  int m = BSIZ / 2;
  b[(m - 1) * BSIZ + m - 1] = WHITE;
  b[(m - 1) * BSIZ + m] = BLACK;
  b[m * BSIZ + m - 1] = BLACK;
  b[m * BSIZ + m] = WHITE;
}

static int valid_at(const char *b, int r, int c, char p) {
  if (r < 0 || r >= BSIZ || c < 0 || c >= BSIZ)
    return 0;
  if (b[r * BSIZ + c] != EMPTY)
    return 0;
  char opp = (p == BLACK) ? WHITE : BLACK;
  for (int d = 0; d < 8; d++) {
    int rr = r + dirs[d][0], cc = c + dirs[d][1], f = 0;
    while (rr >= 0 && rr < BSIZ && cc >= 0 && cc < BSIZ &&
           b[rr * BSIZ + cc] == opp) {
      f = 1;
      rr += dirs[d][0];
      cc += dirs[d][1];
    }
    if (f && rr >= 0 && rr < BSIZ && cc >= 0 && cc < BSIZ &&
        b[rr * BSIZ + cc] == p)
      return 1;
  }
  return 0;
}

static void do_move(char *b, int r, int c, char p) {
  b[r * BSIZ + c] = p;
  char opp = (p == BLACK) ? WHITE : BLACK;
  for (int d = 0; d < 8; d++) {
    int rr = r + dirs[d][0], cc = c + dirs[d][1], n = 0;
    while (rr >= 0 && rr < BSIZ && cc >= 0 && cc < BSIZ &&
           b[rr * BSIZ + cc] == opp) {
      n++;
      rr += dirs[d][0];
      cc += dirs[d][1];
    }
    if (n > 0 && rr >= 0 && rr < BSIZ && cc >= 0 && cc < BSIZ &&
        b[rr * BSIZ + cc] == p) {
      rr = r + dirs[d][0];
      cc = c + dirs[d][1];
      for (int i = 0; i < n; i++) {
        b[rr * BSIZ + cc] = p;
        rr += dirs[d][0];
        cc += dirs[d][1];
      }
    }
  }
}

static void valid_csv(const char *b, char p, char *out, int max) {
  int pos = 0;
  out[0] = '\0';
  for (int r = 0; r < BSIZ; r++)
    for (int c = 0; c < BSIZ; c++)
      if (valid_at(b, r, c, p)) {
        if (pos > 0 && pos < max - 1)
          out[pos++] = ',';
        int n = snprintf(out + pos, (size_t)(max - pos), "%d", r * BSIZ + c);
        if (n > 0)
          pos += n;
      }
}

/* Play one Othello game: model (BLACK) vs random (WHITE). */
static int play_one(const Organelle *org, const MicrogptConfig *cfg,
                    int *errs) {
  char b[CELLS];
  board_init(b);
  char cur = BLACK;
  int passes = 0;

  for (int t = 0; t < 100 && passes < 2; t++) {
    char vs[256];
    valid_csv(b, cur, vs, sizeof(vs));
    if (vs[0] == '\0') {
      passes++;
      cur = (cur == BLACK) ? WHITE : BLACK;
      continue;
    }
    passes = 0;

    if (cur == BLACK) {
      char bs[CELLS + 1];
      memcpy(bs, b, CELLS);
      bs[CELLS] = '\0';
      char prompt[256];
      snprintf(prompt, sizeof(prompt), "board=%s|valid=%s", bs, vs);
      char out[GEN_LEN + 1];
      organelle_generate(org, cfg, prompt, out, GEN_LEN, (scalar_t)TEMP);
      int p = atoi(out);
      int r = p / BSIZ, c = p % BSIZ;
      if (p >= 0 && p < CELLS && valid_at(b, r, c, cur)) {
        do_move(b, r, c, cur);
      } else {
        (*errs)++;
        for (int rr = 0; rr < BSIZ; rr++)
          for (int cc = 0; cc < BSIZ; cc++)
            if (valid_at(b, rr, cc, cur)) {
              do_move(b, rr, cc, cur);
              goto done;
            }
      done:;
      }
    } else {
      int mv[CELLS], nm = 0;
      for (int r = 0; r < BSIZ; r++)
        for (int c = 0; c < BSIZ; c++)
          if (valid_at(b, r, c, cur))
            mv[nm++] = r * BSIZ + c;
      if (nm > 0) {
        int pk = mv[rand() % nm];
        do_move(b, pk / BSIZ, pk % BSIZ, cur);
      }
    }
    cur = (cur == BLACK) ? WHITE : BLACK;
  }

  int bk = 0, wh = 0;
  for (int i = 0; i < CELLS; i++) {
    if (b[i] == BLACK)
      bk++;
    if (b[i] == WHITE)
      wh++;
  }
  return bk > wh;
}

static int eval_model(const char *label, const Organelle *org,
                      const MicrogptConfig *cfg, int ng, int *errs) {
  int w = 0;
  *errs = 0;
  for (int g = 0; g < ng; g++)
    w += play_one(org, cfg, errs);
  printf("  %-22s  %3d/%-3d (%2d%% win)  %d errors\n", label, w, ng,
         w * 100 / ng, *errs);
  return w;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  srand((unsigned)time(NULL));
  seed_rng(42);

  printf("================================================================\n");
  printf("  MicroGPT-C — Transfer Learning Experiment (K-12)\n");
  printf("  Source: TicTacToe Player → Target: Othello Player\n");
  printf(
      "================================================================\n\n");

  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = N_EMBD;
  cfg.n_head = N_HEAD;
  cfg.mlp_dim = MLP_DIM;
  cfg.n_layer = N_LAYER;
  cfg.block_size = BLOCK_SIZE;
  cfg.learning_rate = LEARNING_RATE;
  cfg.batch_size = BATCH_SIZE;
  cfg.max_vocab = MAX_VOCAB;
  cfg.max_docs = MAX_DOCS;
  cfg.max_doc_len = MAX_DOC_LEN;
  microgpt_print_config("Transfer Learning (K-12)", &cfg);

  int ng = 100;

  /* ====== Phase 1: Train TicTacToe source ====== */
  printf("\n--- PHASE 1: Train source (TicTacToe Player) ---\n");
  remove("transfer_ttt.ckpt");
  Organelle *ttt = organelle_train("TTT Source", "tictactoe_player.txt",
                                   "transfer_ttt.ckpt", &cfg, NUM_STEPS);
  if (!ttt) {
    fprintf(stderr, "FATAL: TTT training failed\n");
    return 1;
  }

  /* ====== Phase 2: Train Othello scratch ====== */
  printf("\n--- PHASE 2: Train Othello from scratch (baseline) ---\n");
  remove("transfer_oth.ckpt");
  Organelle *scratch = organelle_train("Othello Scratch", "othello_player.txt",
                                       "transfer_oth.ckpt", &cfg, NUM_STEPS);
  if (!scratch) {
    fprintf(stderr, "FATAL: Othello training failed\n");
    return 1;
  }

  /* ====== Phase 3: Create transfer model ====== */
  printf("\n--- PHASE 3: Transfer TTT internals → Othello model ---\n");

  /* Load Othello corpus just for vocab */
  Docs oth_docs;
  if (opa_load_docs_multiline("othello_player.txt", &oth_docs, cfg.max_docs) !=
      0) {
    fprintf(stderr, "FATAL: othello_player.txt not found\n");
    return 1;
  }
  Vocab oth_vocab;
  build_vocab(&oth_docs, &oth_vocab);
  printf("  Vocab: TTT = %zu chars, Othello = %zu chars\n",
         ttt->vocab.vocab_size, oth_vocab.vocab_size);

  Model *xfer = model_create(oth_vocab.vocab_size, &cfg);
  printf("  Transferring internal transformer weights...\n");
  model_transfer_weights(ttt->model, xfer, &cfg);

  Organelle xfer_org;
  memset(&xfer_org, 0, sizeof(xfer_org));
  xfer_org.model = xfer;
  xfer_org.vocab = oth_vocab;
  xfer_org.docs = oth_docs;

  /* ====== Phase 4: Random untrained baseline ====== */
  printf("\n--- PHASE 4: Create random baseline ---\n");
  Docs rnd_docs;
  opa_load_docs_multiline("othello_player.txt", &rnd_docs, cfg.max_docs);
  Vocab rnd_vocab;
  build_vocab(&rnd_docs, &rnd_vocab);
  Model *rnd = model_create(rnd_vocab.vocab_size, &cfg);
  Organelle rnd_org;
  memset(&rnd_org, 0, sizeof(rnd_org));
  rnd_org.model = rnd;
  rnd_org.vocab = rnd_vocab;
  rnd_org.docs = rnd_docs;

  /* ====== Phase 5: Evaluate ====== */
  printf("\n--- PHASE 5: Playing %d Othello games per condition ---\n\n", ng);

  int e1, e2, e3;
  int w1 = eval_model("SCRATCH (trained)", scratch, &cfg, ng, &e1);
  int w2 = eval_model("TRANSFER (TTT→Oth)", &xfer_org, &cfg, ng, &e2);
  int w3 = eval_model("RANDOM (untrained)", &rnd_org, &cfg, ng, &e3);

  /* ====== Results ====== */
  printf(
      "\n================================================================\n");
  printf("              TRANSFER LEARNING RESULTS\n");
  printf("================================================================\n");
  printf("  Condition             Wins/%-3d  Win%%   Errors\n", ng);
  printf("  ────────────────────  ────────  ─────  ──────\n");
  printf("  SCRATCH (trained)       %3d     %3d%%    %d\n", w1, w1 * 100 / ng,
         e1);
  printf("  TRANSFER (TTT→Oth)      %3d     %3d%%    %d\n", w2, w2 * 100 / ng,
         e2);
  printf("  RANDOM (untrained)      %3d     %3d%%    %d\n", w3, w3 * 100 / ng,
         e3);
  printf(
      "================================================================\n\n");

  if (w2 > w3)
    printf("✅ Transfer > Random: +%d%% (internal representations transfer)\n",
           (w2 - w3) * 100 / ng);
  else
    printf("❌ Transfer ≤ Random: no evidence of representation transfer\n");

  if (w2 > w1)
    printf("✅ Transfer > Scratch: +%d%% (pre-training helps)\n",
           (w2 - w1) * 100 / ng);
  else
    printf("Note: Scratch > Transfer (as expected without fine-tuning)\n");

  /* Cleanup */
  organelle_free(ttt);
  organelle_free(scratch);
  model_free(xfer);
  model_free(rnd);

  return 0;
}
