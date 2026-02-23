/*
 * microgpt_vm_parser.tab.h — Token definitions for the VM parser.
 *
 * Extracted from the Bison-generated microgpt_vm_parser.tab.c so that
 * both the lexer (.l.c) and the main VM source (microgpt_vm.c) can
 * reference token IDs and the semantic value type without pulling in
 * the full parser implementation.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef MICROGPT_VM_PARSER_TAB_H
#define MICROGPT_VM_PARSER_TAB_H

/* vm_module_parser is defined in microgpt_vm.h — include that first */

/* Tokens */
#ifndef VM_MODULE_PARSER_TOKENTYPE
#define VM_MODULE_PARSER_TOKENTYPE
enum vm_module_parser_tokentype {
  NAME = 258,
  NUMBER = 259,
  STRING = 260,
  BOOLEAN = 261,
  COMMENT = 262,
  VAR = 263,
  DECLARE = 264,
  FUNCTION = 265,
  RETURN = 266,
  YIELD = 267,
  IF = 268,
  ELSE = 269,
  CONDITION_GTE = 270,
  CONDITION_LTE = 271,
  CONDITION_GT = 272,
  CONDITION_LT = 273,
  CONDITION_NE = 274,
  CONDITION_EQ = 275,
  CONDITIONAL_AND = 276,
  CONDITIONAL_OR = 277,
  WHILE = 278,
  FOR = 279,
  INCREMENT_BY_ONE = 280,
  DECREMENT_BY_ONE = 281,
  INCREMENT_BY = 282,
  DECREMENT_BY = 283,
  OPERATOR_SUM = 284,
  OPERATOR_FRAC = 285,
  OPERATOR_POW = 286,
  UMINUS = 287
};
#endif

/* Semantic value type */
#ifndef VM_MODULE_PARSER_STYPE_IS_DECLARED
#define VM_MODULE_PARSER_STYPE_IS_DECLARED
typedef union VM_MODULE_PARSER_STYPE {
  char *string;
} VM_MODULE_PARSER_STYPE;
#define VM_MODULE_PARSER_STYPE_IS_TRIVIAL 1
#define vm_module_parser_stype VM_MODULE_PARSER_STYPE
#endif

/* Substitute names used by Bison */
#define YYSTYPE VM_MODULE_PARSER_STYPE

extern VM_MODULE_PARSER_STYPE vm_module_parser_lval;

int vm_module_parser_parse(vm_module_parser *parser);

#endif /* MICROGPT_VM_PARSER_TAB_H */
