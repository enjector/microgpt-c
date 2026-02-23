/*
 * MicroGPT-C VM Engine — Flex Lexer
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * microgpt_vm.l  —  Flex lexer for the Virtual Machine parser.
 *
 * Tokenises a TypeScript-like language plus LaTeX mathematical notation.
 * Uses custom YY_INPUT to read from an in-memory string buffer (no files).
 * All tokens use the "vm_module_parser_" prefix (%option prefix).
 *
 * Copyright 2024 Enjector Software, Ltd.  MIT License.
 */

%option prefix="vm_module_parser_"
%option nounistd

%{
	#include "microgpt_vm.h"

	#include <stdlib.h>
	#include <string.h>

#ifndef _WIN32
	#include <unistd.h>
	#else
	#include <io.h>
	#define isatty _isatty
	#endif

	#ifdef __APPLE__
	#include <unistd.h>
	#endif
	
	/* Bison-generated token definitions */
	#include "microgpt_vm_parser.tab.h"

	extern int vm_module_parser_char_fetch_next(vm_module_parser* parser);
	extern vm_module_parser* _vm_ctx_current_parser;
	
	/*
	 * Custom input: read one character at a time from the in-memory
	 * source string instead of from a FILE*.
	 */
	#undef YY_INPUT
	#define YY_INPUT(buf, result, max_size) \
	{ \
		int c = vm_module_parser_char_fetch_next(_vm_ctx_current_parser); \
		result = (c == 0) ? YY_NULL : (buf[0] = c, 1); \
		if (result != YY_NULL) { ++_vm_ctx_current_parser->vm_module_parser_state_input_line_column; } \
	}

	#define YY_DECL int yylex(vm_module_parser* parser)
%}

/* ── Pattern definitions ──────────────────────────────────────────────── */

nl				(\r\n|\r|\n)
ws				[ \t]+
name			[a-zA-Z_][0-9a-zA-Z_]*
number			[0-9][0-9]*(\.[0-9]+)?
comment			("//".*)
block_comment		([/][*][^*]*[*]+([^*/][^*]*[*]+)*[/])
string			(\'(\\.|[^'])*\'|\"(\\.|[^"])*\")

/* Comparison and logical operators */
condition_gte		">="
condition_lte		"<="
condition_gt		">"
condition_lt		"<"
condition_ne		"!="
condition_eq		"=="
conditional_and		"&&"
conditional_or		"||"

/* Increment / decrement operators */
increment_by_one	"++"
decrement_by_one	"--"
increment_by		"+="
decrement_by		"-="

/* Greek variable names (UTF-8 or LaTeX escape) */
symbol_alpha		("α"|"\\alpha")

/* LaTeX math operators */
operator_sum		"\\sum_"
operator_frac		"\\frac"
operator_pow		"\\pow"

%%


function			{ return FUNCTION; }
declare				{ return DECLARE; }
return				{ return RETURN; }
if					{ return IF; }
else				{ return ELSE; }
yield				{ return YIELD; }
var					{ return VAR; }
while				{ return WHILE; }
for					{ return FOR; }


true				{ vm_module_parser_lval.string = string_clone("true"); return BOOLEAN; }
false				{ vm_module_parser_lval.string = string_clone("false"); return BOOLEAN; }

{string}			{	size_t yystring_len = strlen(yytext);
						/* Allocate space for content (minus quotes) + null terminator */
						vm_module_parser_lval.string = (char*) malloc(yystring_len - 1);
						memcpy(vm_module_parser_lval.string, &yytext[1], yystring_len - 2);
						vm_module_parser_lval.string[yystring_len - 2] = '\0';
						return STRING; 
					}

{name}				{ vm_module_parser_lval.string = string_clone(yytext); return NAME; }
{number}			{ vm_module_parser_lval.string = string_clone(yytext); return NUMBER; }


{condition_gte}		{ return CONDITION_GTE; }
{condition_lte}		{ return CONDITION_LTE; }
{condition_gt}		{ return CONDITION_GT; }
{condition_lt}		{ return CONDITION_LT; }
{condition_ne}		{ return CONDITION_NE; }
{condition_eq}		{ return CONDITION_EQ; }
{conditional_and}	{ return CONDITIONAL_AND; }
{conditional_or}	{ return CONDITIONAL_OR; }


{increment_by_one}	{ return INCREMENT_BY_ONE; }
{decrement_by_one}	{ return DECREMENT_BY_ONE; }
{increment_by}		{ return INCREMENT_BY; }
{decrement_by}		{ return DECREMENT_BY; }


{symbol_alpha}      { vm_module_parser_lval.string = string_clone("alpha"); return NAME; }


{operator_sum}		{ return OPERATOR_SUM; }
{operator_frac}		{ return OPERATOR_FRAC; }
{operator_pow}		{ return OPERATOR_POW; }


{comment}			{ vm_module_parser_lval.string = yytext; return COMMENT; }
{block_comment}		{ vm_module_parser_lval.string = yytext; return COMMENT; }

{ws}				{/* skip whitespace */}
{nl}				{ parser->vm_module_parser_state_input_line_number++; parser->vm_module_parser_state_input_line_column = 1; }


.					{ return yytext[0]; }
