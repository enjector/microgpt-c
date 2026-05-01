/*
 * microgpt_pipeline_vm.c — Opt-in VM-backed dispatcher for the Pipeline IR.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 *
 * This translation unit implements pipeline_execute_vm() — a convenience
 * dispatcher that resolves each leaf primitive name to a vm_native_fn
 * registered in a vm_engine via vm_engine_find_fn(), marshals
 * PipelineValue ↔ double, and dispatches.
 *
 * Why this lives in a separate TU:
 *   microgpt_lib.a does NOT link the VM (it is a heavy module — ~4 K LOC).
 *   This TU is opt-in: a demo or test that calls pipeline_execute_vm()
 *   must add BOTH microgpt_pipeline_vm.c AND microgpt_vm.c to its target
 *   sources.  microgpt_lib.a is unaware of the VM, keeping its footprint
 *   minimal for the many demos that do not use VM dispatch.
 *
 * Constraints (per BS_pipeline_ir.md INV-PIPE-024):
 *   - INT/FLOAT/VOID ports only.  STRING/LIST/TENSOR/RECORD ports cause
 *     PIPE_ERR_EXEC with a message that identifies the offending node
 *     and port (closes GAP-PIPE-002).
 *   - Subgraph nodes recurse via pipeline_execute_vm.
 *   - Native ABI is `double(int argc, const double *argv)` — single
 *     numeric return.  Leaf nodes with multiple outputs receive the
 *     native's return value on the FIRST output port; any additional
 *     output ports are filled with 0.0.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"
#include "microgpt_pipeline_internal.h"
#include "microgpt_vm.h"

#include <stdint.h>
#include <stdlib.h>

int pipeline_execute_vm(const Pipeline *p,
                        vm_engine *vm,
                        const PipelineValue *inputs,
                        PipelineValue *outputs) {
    if (!p) {
        mgpt_pipe_set_err("pipeline_execute_vm: null pipeline");
        return PIPE_ERR_EXEC;
    }
    if (!vm) {
        mgpt_pipe_set_err("pipeline_execute_vm: null vm_engine");
        return PIPE_ERR_EXEC;
    }
    if (!p->verified) {
        mgpt_pipe_set_err("pipeline_execute_vm: graph not verified");
        return PIPE_ERR_EXEC;
    }

    PipelineValue *edge_values =
        (PipelineValue *)calloc(p->n_edges + 1, sizeof(PipelineValue));
    if (!edge_values) {
        mgpt_pipe_set_err("OOM");
        return PIPE_ERR_OOM;
    }

    /* Pre-fill edges from signature inputs. */
    for (size_t e = 0; e < p->n_edges; e++) {
        const PipelineEdge *edge = p->edges[e];
        if (edge->src_node_idx == MGPT_PIPE_SIG_IN_NODE) {
            int si = edge->src_port_idx;
            edge_values[e] = inputs[si];
            edge_values[e].type = p->signature_in[si].type;
        }
    }

    int rc = PIPE_OK;
    for (size_t k = 0; k < p->n_nodes; k++) {
        int n_idx = p->exec_order[k];
        PipelineNode *node = p->nodes[n_idx];

        PipelineValue *in_vals = (PipelineValue *)calloc(
            (size_t)(node->n_inputs > 0 ? node->n_inputs : 1),
            sizeof(PipelineValue));
        PipelineValue *out_vals = (PipelineValue *)calloc(
            (size_t)(node->n_outputs > 0 ? node->n_outputs : 1),
            sizeof(PipelineValue));
        if (!in_vals || !out_vals) {
            free(in_vals);
            free(out_vals);
            rc = PIPE_ERR_OOM;
            goto cleanup;
        }

        for (int ip = 0; ip < node->n_inputs; ip++) {
            int eidx = mgpt_pipe_find_incoming_edge(p, n_idx, ip);
            if (eidx < 0) {
                free(in_vals);
                free(out_vals);
                mgpt_pipe_set_err(
                    "pipeline_execute_vm: missing edge for node '%s' input '%s'",
                    node->id ? node->id : "?",
                    node->inputs[ip].name ? node->inputs[ip].name : "?");
                rc = PIPE_ERR_EXEC;
                goto cleanup;
            }
            in_vals[ip] = edge_values[eidx];
            in_vals[ip].type = node->inputs[ip].type;
        }
        for (int op = 0; op < node->n_outputs; op++) {
            out_vals[op].type = node->outputs[op].type;
        }

        if (node->subgraph) {
            int sub_rc = pipeline_execute_vm(node->subgraph, vm, in_vals, out_vals);
            if (sub_rc != 0) {
                free(in_vals);
                free(out_vals);
                rc = sub_rc;
                goto cleanup;
            }
        } else {
            /* Type-check ports — INT/FLOAT/VOID only. */
            for (int ip = 0; ip < node->n_inputs; ip++) {
                PipelineTypeKind k_in = node->inputs[ip].type
                                            ? node->inputs[ip].type->kind
                                            : PIPE_T_VOID;
                if (k_in != PIPE_T_INT && k_in != PIPE_T_FLOAT &&
                    k_in != PIPE_T_VOID) {
                    free(in_vals);
                    free(out_vals);
                    mgpt_pipe_set_err(
                        "pipeline_execute_vm: node '%s' input port '%s' has "
                        "non-numeric type (STRING/LIST/TENSOR/RECORD); only "
                        "INT/FLOAT/VOID supported by the VM ABI",
                        node->id ? node->id : "?",
                        node->inputs[ip].name ? node->inputs[ip].name : "?");
                    rc = PIPE_ERR_EXEC;
                    goto cleanup;
                }
            }
            for (int op = 0; op < node->n_outputs; op++) {
                PipelineTypeKind k_out = node->outputs[op].type
                                             ? node->outputs[op].type->kind
                                             : PIPE_T_VOID;
                if (k_out != PIPE_T_INT && k_out != PIPE_T_FLOAT &&
                    k_out != PIPE_T_VOID) {
                    free(in_vals);
                    free(out_vals);
                    mgpt_pipe_set_err(
                        "pipeline_execute_vm: node '%s' output port '%s' has "
                        "non-numeric type (STRING/LIST/TENSOR/RECORD); only "
                        "INT/FLOAT/VOID supported by the VM ABI",
                        node->id ? node->id : "?",
                        node->outputs[op].name ? node->outputs[op].name : "?");
                    rc = PIPE_ERR_EXEC;
                    goto cleanup;
                }
            }

            vm_native_fn fn = vm_engine_find_fn(vm, node->primitive);
            if (!fn) {
                free(in_vals);
                free(out_vals);
                mgpt_pipe_set_err(
                    "pipeline_execute_vm: native function '%s' (node '%s') "
                    "not registered in vm_engine; call vm_engine_register_fn() "
                    "before pipeline_execute_vm",
                    node->primitive ? node->primitive : "?",
                    node->id ? node->id : "?");
                rc = PIPE_ERR_EXEC;
                goto cleanup;
            }

            double argv[32];
            int argc = node->n_inputs;
            if (argc > 32) {
                free(in_vals);
                free(out_vals);
                mgpt_pipe_set_err(
                    "pipeline_execute_vm: node '%s' has %d inputs, exceeds "
                    "32-arg native ABI limit",
                    node->id ? node->id : "?", argc);
                rc = PIPE_ERR_EXEC;
                goto cleanup;
            }
            for (int ip = 0; ip < argc; ip++) {
                PipelineTypeKind k_in = in_vals[ip].type
                                            ? in_vals[ip].type->kind
                                            : PIPE_T_VOID;
                if (k_in == PIPE_T_INT)
                    argv[ip] = (double)in_vals[ip].v.i;
                else if (k_in == PIPE_T_FLOAT)
                    argv[ip] = in_vals[ip].v.f;
                else
                    argv[ip] = 0.0;
            }

            double ret = fn(argc, argv);

            for (int op = 0; op < node->n_outputs; op++) {
                PipelineTypeKind k_out = out_vals[op].type
                                             ? out_vals[op].type->kind
                                             : PIPE_T_VOID;
                double v = (op == 0) ? ret : 0.0;
                if (k_out == PIPE_T_INT)
                    out_vals[op].v.i = (int64_t)v;
                else if (k_out == PIPE_T_FLOAT)
                    out_vals[op].v.f = v;
                /* VOID: leave zero-initialised */
            }
        }

        for (size_t e = 0; e < p->n_edges; e++) {
            const PipelineEdge *edge = p->edges[e];
            if (edge->src_node_idx != n_idx)
                continue;
            edge_values[e] = out_vals[edge->src_port_idx];
        }
        free(in_vals);
        free(out_vals);
    }

    for (int so = 0; so < p->n_sig_out; so++) {
        int eidx = -1;
        for (size_t e = 0; e < p->n_edges; e++) {
            if (p->edges[e]->dst_node_idx == MGPT_PIPE_SIG_OUT_NODE &&
                p->edges[e]->dst_port_idx == so) {
                eidx = (int)e;
                break;
            }
        }
        if (eidx < 0) {
            mgpt_pipe_set_err(
                "pipeline_execute_vm: signature output '%s' unconnected",
                p->signature_out[so].name ? p->signature_out[so].name : "?");
            rc = PIPE_ERR_EXEC;
            goto cleanup;
        }
        outputs[so] = edge_values[eidx];
        outputs[so].type = p->signature_out[so].type;
    }

cleanup:
    free(edge_values);
    return rc;
}
