/*
 * tictactoe_transfer_planner_to_player — Same-Game Transfer Learning (K-14)
 *
 * Tests whether representations from a TicTacToe Planner transfer to
 * a TicTacToe Player.  Since both organelles operate on the same game,
 * their vocabularies partially overlap (board chars, digits, pipe).
 *
 * Three conditions:
 *   1. SCRATCH     — TTT Player trained from scratch (baseline)
 *   2. TRANSFER+FT — Planner internals transferred, fine-tuned on Player corpus
 *   3. RANDOM      — Untrained model (negative control)
 */

#include "microgpt.h"
#include "microgpt_organelle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- TTT game logic ---- */
#define EMPTY '_'
#define PX 'X'
#define PO 'O'
#define TEMP 0.3
#define GEN_LEN 12

static char check_winner(const char *b) {
  static const int lines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                                  {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
  for (int i = 0; i < 8; i++) {
    int a = lines[i][0], b_ = lines[i][1], c = lines[i][2];
    if (b[a] != EMPTY && b[a] == b[b_] && b[b_] == b[c])
      return b[a];
  }
  return EMPTY;
}

static int board_full(const char *b) {
  for (int i = 0; i < 9; i++)
    if (b[i] == EMPTY)
      return 0;
  return 1;
}

static void valid_csv(const char *b, char *out, int max) {
  int pos = 0;
  out[0] = '\0';
  for (int i = 0; i < 9; i++) {
    if (b[i] == EMPTY) {
      if (pos > 0 && pos < max - 1)
        out[pos++] = ',';
      int n = snprintf(out + pos, (size_t)(max - pos), "%d", i);
      if (n > 0)
        pos += n;
    }
  }
}

/* Play one TTT game: model (X) vs random (O). Returns 1 if X wins. */
static int play_one(const Organelle *org, const MicrogptConfig *cfg,
                    int *errs) {
  char b[10];
  memset(b, EMPTY, 9);
  b[9] = '\0';
  char cur = PX;

  for (int t = 0; t < 9; t++) {
    if (board_full(b) || check_winner(b) != EMPTY)
      break;

    if (cur == PX) {
      char vs[32];
      valid_csv(b, vs, sizeof(vs));
      char prompt[64];
      snprintf(prompt, sizeof(prompt), "board=%s|valid=%s", b, vs);
      char out[GEN_LEN + 1];
      organelle_generate(org, cfg, prompt, out, GEN_LEN, (scalar_t)TEMP);
      int p = -1;
      if (out[0] >= '0' && out[0] <= '8')
        p = out[0] - '0';
      if (p >= 0 && p < 9 && b[p] == EMPTY) {
        b[p] = PX;
      } else {
        (*errs)++;
        for (int i = 0; i < 9; i++)
          if (b[i] == EMPTY) {
            b[i] = PX;
            break;
          }
      }
    } else {
      /* Random opponent */
      int emp[9], ne = 0;
      for (int i = 0; i < 9; i++)
        if (b[i] == EMPTY)
          emp[ne++] = i;
      if (ne > 0)
        b[emp[rand() % ne]] = PO;
    }
    cur = (cur == PX) ? PO : PX;
  }

  return check_winner(b) == PX;
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
  printf("  MicroGPT-C — Same-Game Transfer (K-14)\n");
  printf("  TTT Planner → TTT Player\n");
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
  microgpt_print_config("Same-Game Transfer (K-14)", &cfg);

  int ng = 100;

  /* ====== Phase 1: Train TTT Planner (source) ====== */
  printf("\n--- PHASE 1: Train source (TTT Planner) ---\n");
  remove("c_transfer_planner.ckpt");
  Organelle *planner = organelle_train("TTT Planner", "c_tictactoe_planner.txt",
                                       "c_transfer_planner.ckpt", &cfg, NUM_STEPS);
  if (!planner) {
    fprintf(stderr, "FATAL: Planner training failed\n");
    return 1;
  }

  /* ====== Phase 2: Train TTT Player from scratch (baseline) ====== */
  printf("\n--- PHASE 2: Train TTT Player from scratch ---\n");
  remove("c_transfer_scratch.ckpt");
  Organelle *scratch = organelle_train("Player Scratch", "c_tictactoe_player.txt",
                                       "c_transfer_scratch.ckpt", &cfg, NUM_STEPS);
  if (!scratch) {
    fprintf(stderr, "FATAL: Player training failed\n");
    return 1;
  }

  /* ====== Phase 3: Train TTT Player with Planner transfer+FT ====== */
  printf("\n--- PHASE 3: Transfer Planner → Player + Fine-tune ---\n");
  remove("c_transfer_ft.ckpt");
  Organelle *xfer_ft = organelle_train_transfer(
      "Player Transfer+FT", "c_tictactoe_player.txt", "c_transfer_ft.ckpt", &cfg,
      NUM_STEPS, planner->model);
  if (!xfer_ft) {
    fprintf(stderr, "FATAL: Transfer+FT training failed\n");
    return 1;
  }

  /* ====== Phase 4: Random untrained baseline ====== */
  printf("\n--- PHASE 4: Create random baseline ---\n");
  Docs rnd_docs;
  opa_load_docs_multiline("c_tictactoe_player.txt", &rnd_docs, cfg.max_docs);
  Vocab rnd_vocab;
  build_vocab(&rnd_docs, &rnd_vocab);
  Model *rnd = model_create(rnd_vocab.vocab_size, &cfg);
  Organelle rnd_org;
  memset(&rnd_org, 0, sizeof(rnd_org));
  rnd_org.model = rnd;
  rnd_org.vocab = rnd_vocab;
  rnd_org.docs = rnd_docs;

  /* ====== Phase 5: Evaluate ====== */
  printf("\n--- PHASE 5: Playing %d TTT games per condition ---\n\n", ng);

  int e1, e2, e3;
  int w1 = eval_model("SCRATCH (trained)", scratch, &cfg, ng, &e1);
  int w2 = eval_model("TRANSFER+FT", xfer_ft, &cfg, ng, &e2);
  int w3 = eval_model("RANDOM (untrained)", &rnd_org, &cfg, ng, &e3);

  /* Results */
  printf(
      "\n================================================================\n");
  printf("              SAME-GAME TRANSFER RESULTS\n");
  printf("================================================================\n");
  printf("  Condition             Wins/%-3d  Win%%   Errors\n", ng);
  printf("  ──────────────────────  ────────  ─────  ──────\n");
  printf("  SCRATCH (trained)       %3d     %3d%%    %d\n", w1, w1 * 100 / ng,
         e1);
  printf("  TRANSFER+FT             %3d     %3d%%    %d\n", w2, w2 * 100 / ng,
         e2);
  printf("  RANDOM (untrained)      %3d     %3d%%    %d\n", w3, w3 * 100 / ng,
         e3);
  printf(
      "================================================================\n\n");

  /* Analysis */
  if (w2 > w1)
    printf("✅ Transfer+FT > Scratch: +%d%% (planner knowledge transfers!)\n",
           (w2 - w1) * 100 / ng);
  else if (w2 > w3)
    printf("✅ Transfer+FT > Random: +%d%% (some benefit from transfer)\n",
           (w2 - w3) * 100 / ng);
  else
    printf("❌ Transfer+FT ≤ Random: no evidence of same-game transfer\n");

  printf("\nVocab overlap note:\n");
  printf("  Planner vocab: %zu chars, Player vocab: %zu chars\n",
         planner->vocab.vocab_size, scratch->vocab.vocab_size);

  /* Cleanup */
  organelle_free(planner);
  organelle_free(scratch);
  organelle_free(xfer_ft);
  model_free(rnd);

  return 0;
}
