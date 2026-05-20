/*
 * src/microgpt_pipeline.h — Legacy compatibility shim.
 *
 * The canonical Pipeline IR public header now lives at
 *   libs/pipeline_ir/include/pipeline_ir/pipeline_ir.h
 *
 * It was extracted from this tree by Experiment E02 (see
 * experiments/E02-pipeline-ir-library.md) so external consumers can
 * adopt the IR + verifier + DOT renderer without depending on
 * MicroGPT-C's transformer engine.
 *
 * Every demo, test, and tool inside this repo historically included
 * "microgpt_pipeline.h"; rather than rewrite every caller, this shim
 * forwards to the new public header.  New external code should
 * include <pipeline_ir/pipeline_ir.h> directly.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */
#ifndef MICROGPT_PIPELINE_LEGACY_SHIM_H
#define MICROGPT_PIPELINE_LEGACY_SHIM_H

#include "pipeline_ir/pipeline_ir.h"

#endif /* MICROGPT_PIPELINE_LEGACY_SHIM_H */
