/*
 * microgpt_pipeline_internal.h — Private helpers shared between
 *   microgpt_pipeline.c (core IR + verifier + callback executor)
 * and
 *   microgpt_pipeline_vm.c  (VM-backed dispatcher, opt-in module)
 *
 * NOT a public API.  Do NOT include from user code; do NOT install.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef MICROGPT_PIPELINE_INTERNAL_H
#define MICROGPT_PIPELINE_INTERNAL_H

#include "microgpt_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel "node indices" for signature ports on edges. */
#define MGPT_PIPE_SIG_IN_NODE  (-1)
#define MGPT_PIPE_SIG_OUT_NODE (-2)

/* Set the thread-local last-error message (printf-style).
 * Implemented in microgpt_pipeline.c. */
void mgpt_pipe_set_err(const char *fmt, ...);

/* Find the unique incoming edge for (dst_node, dst_port). Returns the edge
 * index in p->edges[], or -1 if no such edge.  Implemented in
 * microgpt_pipeline.c. */
int mgpt_pipe_find_incoming_edge(const Pipeline *p, int dst_node, int dst_port);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_PIPELINE_INTERNAL_H */
