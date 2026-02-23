/*
 * MicroGPT-C VM Engine — Bison Grammar
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * microgpt_vm.y  —  Bison grammar for the Virtual Machine parser.
 *
 * Defines the TypeScript-like language plus LaTeX mathematical notation.
 * Grammar actions directly emit bytecode via generator macros (no AST).
 * Uses "vm_module_parser_" prefix for all Bison symbols.
 *
 * Copyright 2024 Enjector Software, Ltd.  MIT License.
 */

%define api.prefix {vm_module_parser_}

%{
	#include "microgpt_vm.h"
	#include <stdio.h>

	extern void vm_module_parser_error(vm_module_parser* parser, char* error);
	extern int yylex(vm_module_parser* parser);

	/*
	 * Temporary globals used by grammar actions.
	 * WARNING: These make the parser non-reentrant / non-thread-safe.
	 */
	char* vm_tmp_var = 0;
	char* vm_tmp_label = 0;
	char* vm_tmp_type = 0;
	queue* tracking_labels;

	/* ── Code generation shorthand macros ─────────────────────────────── */

	#define function_begin(name)						vm_module_generator_function_begin(parser->generator, name)
	#define function_end()								vm_module_generator_function_end(parser->generator)
	#define function_parameter(name, type)				vm_module_generator_function_parameter(parser->generator, name, type)
	#define function_return_type_set(name, type)		vm_module_generator_function_return_type_set(parser->generator, name, (char*) type)
	#define function_return_type_get(name)				vm_module_generator_function_return_type_get(parser->generator, name)
	#define emit(opcode, param1, param2, param3)		vm_module_generator_function_emit_with_meta(parser->generator, opcode, (char*) param1, (char*) param2, (char*) param3, (char*) parser->vm_module_parser_state_input, parser->vm_module_parser_state_input_line_number, parser->vm_module_parser_state_input_index)
	#define emit_defer_push_begin()						vm_module_generator_defer_push_begin(parser->generator)
	#define emit_defer_push_end()						vm_module_generator_defer_push_end(parser->generator)
	#define emit_defer_pop()							vm_module_generator_defer_pop(parser->generator)
	#define create_register()							vm_module_generator_tmp_register_create(parser->generator)
	#define reset_registers()							vm_module_generator_tmp_registers_reset(parser->generator)
	#define create_label()								vm_module_generator_tmp_label_create(parser->generator)
	#define labels_pop()								vm_module_generator_tracking_labels_pop(parser->generator)
	#define labels_push(symbol)							vm_module_generator_tracking_labels_push(parser->generator, symbol)
	#define track(symbol)								vm_module_generator_symbol_track(parser->generator, symbol)
	#define trait_type_set(symbol, type, is_constant)	vm_module_generator_trait_type_set(parser->generator, symbol, type, is_constant)
	#define trait_type_get(symbol)						vm_module_generator_trait_type_get(parser->generator, symbol)
	#define trait_type_link(source, target) 			trait_type_set(target, trait_type_get(source), false); trait_type_set(source, trait_type_get(target), false);
    #define comment(message)							vm_module_generator_function_emit_comment_with_meta(parser->generator, (char*) message, (char*) parser->vm_module_parser_state_input, parser->vm_module_parser_state_input_line_number, parser->vm_module_parser_state_input_index)
%}

%parse-param { vm_module_parser* parser }
%lex-param { vm_module_parser* parser }

/*
 * Declare the type of values in the grammar
 */

%union
{
	char* string;
}

/*
 * Token types: These are returned by the lexer
 */

%token <string> NAME NUMBER STRING BOOLEAN COMMENT
%token VAR DECLARE
%token FUNCTION RETURN YIELD
%token IF ELSE 
%token CONDITION_GTE CONDITION_LTE CONDITION_GT CONDITION_LT CONDITION_NE CONDITION_EQ
%token CONDITIONAL_AND CONDITIONAL_OR
%token WHILE FOR
%token INCREMENT_BY_ONE DECREMENT_BY_ONE INCREMENT_BY DECREMENT_BY
%token OPERATOR_SUM OPERATOR_FRAC OPERATOR_POW

%left '-' '+'
%left '*' '/'
%left '^'
%nonassoc UMINUS

/*
 * These are production names:
 */

%type <string> code
%type <string> var_declaration
%type <string> function function_name function_call function_header function_parameters function_parameter declare_function_header declare_function_parameter expression_params
%type <string> statements statement
%type <string> if_condition if_condition_else
%type <string> while while_statement
%type <string> for for_statement
%type <string> conditional_expressions conditional_expression
%type <string> expression math_expression
%type <string> latex_math_function_header latex_math_function_name latex_math_function_parameter latex_math_expression
%type <string> opsum_expression opsum

%%

/* ═════════════════════════════════════════════════════════════════════════
 *  Top-level: a source file is comments + function definitions
 * ═════════════════════════════════════════════════════════════════════════
 */
code
: comments functions comments					{}
;

comments
: comments comment 		{}
|						{}
;

comment
: COMMENT				{ comment($1); }

/* ═════════════════════════════════════════════════════════════════════════
 *  Function definitions (TypeScript-style and LaTeX math)
 * ═════════════════════════════════════════════════════════════════════════
 */
functions
: functions function			{}
|								{}
;

/* End of the new function. This will cause a new Function to be created
 */
function
: comments function_header '{' statements '}'	{ function_end(); }
| declare_function_header ';' 					{ emit(opCALL_EXT_METHOD, 0, 0, 0); function_end(); }
| latex_math_function ';'						{}
| latex_math_function 							{}
;

function_header
: function_name '(' function_parameters ')' ':' NAME	{ function_return_type_set($1, $6); trait_type_set($1, $6, false); track($6); $$ = $1; }
| function_name '(' function_parameters ')'				{ vm_tmp_type = string_clone("void"); function_return_type_set($1, vm_tmp_type); trait_type_set($1, vm_tmp_type, false); track(vm_tmp_type); $$ = $1; }
;

declare_function_header
: DECLARE function_name '(' declare_function_parameters ')' ':' NAME	{ function_return_type_set($2, $7); trait_type_set($2, $7, false); track($7); $$ = $2; }
| DECLARE function_name '(' declare_function_parameters ')'				{ function_return_type_set($2, track(string_clone("void"))); trait_type_set($2, string_clone("void"), false); $$ = $2; }
;

/* Create a new function
 */
function_name
: FUNCTION NAME 		{ reset_registers(); function_begin($2); $$ = $2; }
;

/* To get function parameters to populate they are poped from the stack and
 * placed into the variable names which is the parameter name
 */
function_parameters
: function_parameters ',' function_parameter	{ }
| function_parameter							{ }
|												{ }
;

function_parameter
: NAME ':' NAME								{ function_parameter($1, $3); emit(opSTACK_POP, 0, 0, $1); trait_type_set($1, $3, false); track($1); track($3); $$ = $1; }
;

declare_function_parameters
: declare_function_parameters ',' declare_function_parameter	{ }
| declare_function_parameter									{ }
|																{ }
;

declare_function_parameter
: NAME ':' NAME								{ function_parameter($1, $3); trait_type_set($1, $3, false); track($1); track($3); $$ = $1; }
;

/* ═════════════════════════════════════════════════════════════════════════
 *  Statements (assignment, control flow, function calls)
 * ═════════════════════════════════════════════════════════════════════════
 */
statements
: statements statement	{}
|						{}
;

/* A statement can be one of the following:
 *	<variable name> = <expression>					assigns a variable to the value of an expression
 *	<object name>.<property name> = <expression>	assigns the property of an object to the value of an expression
 *	<function name>([<parameters>])					calls a function with optional parameters
 */
statement
: var_declaration ';'							{ }
| NAME '=' expression ';'						{ emit(opSET_VAR, $3, 0, $1); trait_type_link($3, $1); track($1); }
| NAME '=' conditional_expressions ';'			{ emit(opSET_VAR, $3, 0, $1); trait_type_link($3, $1); track($1); }
| NAME '.' NAME '=' expression ';'				{ emit(opSET_OBJ_VAR, $1, $3, $5); trait_type_link($5, $3); /* TODO */ }
| NAME '.' NAME '=' conditional_expressions ';'	{ emit(opSET_OBJ_VAR, $1, $3, $5); trait_type_link($5, $3); /* TODO */}
| increment_expression ';'						{ }
| function_call	';'								{ }
| RETURN expression ';'							{ emit(opSTACK_PUSH, $2, 0, 0); emit(opRETURN, 0, 0, 0); }
| RETURN conditional_expressions ';'			{ emit(opSTACK_PUSH, $2, 0, 0); emit(opRETURN, 0, 0, 0); }
| RETURN ';'									{ emit(opRETURN, 0, 0, 0);  }
| YIELD '(' NAME ')' ';'						{ emit(opYIELD, $3, 0, 0); track($3); }
| while_statement '{' statements '}'			{ 
	/* loop */
	comment("while_loop_end");
	vm_tmp_label = labels_pop(); emit(opJUMP, 0, 0, labels_pop()); emit(opLABEL, vm_tmp_label, 0, 0); }
| for_statement '{' statements '}'				{ 
	/* for_increment_expression */
	emit_defer_pop();
	/* loop */
	comment("for_loop_end");
	vm_tmp_label = labels_pop(); emit(opJUMP, 0, 0, labels_pop()); emit(opLABEL, vm_tmp_label, 0, 0); 
	}
| if_condition '{' statements '}' 				{ emit(opLABEL, labels_pop(), 0, 0); }
| if_condition_else '{' statements '}'			{ emit(opLABEL, labels_pop(), 0, 0); }
| if_condition_else	statement					{ emit(opLABEL, labels_pop(), 0, 0); }
| if_condition statement						{ emit(opLABEL, labels_pop(), 0, 0); }
| COMMENT										{ comment($1); }
;

increment_expression
: NAME INCREMENT_BY_ONE	       					{ emit(opINC, $1, 0, 0); track($1); }
| NAME DECREMENT_BY_ONE	       					{ emit(opDEC, $1, 0, 0); track($1); }
| NAME INCREMENT_BY	expression        			{ emit(opADD, $1, $3, $1); trait_type_link($3, $1); track($1); }
| NAME DECREMENT_BY	expression       			{ emit(opSUB, $1, $3, $1); trait_type_link($3, $1); track($1);  }
;

var_declaration
: VAR NAME '=' expression 						{ emit(opCREATE_SET_VAR, $4, 0, $2); trait_type_link($4, $2); track($2); }
| VAR NAME '=' conditional_expressions 			{ emit(opCREATE_SET_VAR, $4, 0, $2); trait_type_link($4, $2); track($2); }
;

if_condition
: IF '(' conditional_expressions ')'			{ vm_tmp_label = create_label(); emit(opJUMP_IF_FALSE, $3, 0, vm_tmp_label); labels_push(vm_tmp_label); }
;

if_condition_else
: if_condition '{' statements '}' ELSE			{ vm_tmp_label = create_label(); emit(opJUMP, 0, 0, vm_tmp_label); emit(opLABEL, labels_pop(), 0, 0); labels_push(vm_tmp_label); }
;

while_statement
: while '(' conditional_expressions ')'			{ vm_tmp_label = create_label(); emit(opJUMP_IF_FALSE, $3, 0, vm_tmp_label); labels_push(vm_tmp_label); }
;

while
: WHILE											{ 	comment("while_loop_begin");
													vm_tmp_label = create_label(); emit(opLABEL, vm_tmp_label, 0, 0); labels_push(vm_tmp_label); 
												}
;

for_statement
: for for_conditional_loop ';' for_increment_expression ')' {  }
;

for
: FOR '(' var_declaration ';'							{ 	comment("for_loop_begin");
															vm_tmp_label = create_label(); emit(opLABEL, vm_tmp_label, 0, 0); labels_push(vm_tmp_label); 
														}
;

for_conditional_loop
: conditional_expressions 								{  	vm_tmp_label = create_label(); emit(opJUMP_IF_FALSE, $1, 0, vm_tmp_label); labels_push(vm_tmp_label);
															emit_defer_push_begin(); 
															comment("for_increment_expression");
														}
;

for_increment_expression
: increment_expression									{ 	emit_defer_push_end();}
;

conditional_expressions
: '(' conditional_expressions ')'									{ $$ = $2; }
| conditional_expressions CONDITIONAL_AND conditional_expression	{ vm_tmp_var = create_register(); emit(opCONDITIONAL_AND, $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| conditional_expressions CONDITIONAL_OR conditional_expression		{ vm_tmp_var = create_register(); emit(opCONDITIONAL_OR,  $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| conditional_expression											{ $$ = $1; }
;

conditional_expression
: expression CONDITION_GTE expression	{ vm_tmp_var = create_register(); emit(opCONDITION_GTE,   $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression CONDITION_LTE expression	{ vm_tmp_var = create_register(); emit(opCONDITION_LTE,   $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression CONDITION_GT expression	{ vm_tmp_var = create_register(); emit(opCONDITION_GT,    $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression CONDITION_LT expression	{ vm_tmp_var = create_register(); emit(opCONDITION_LT,    $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression CONDITION_NE expression	{ vm_tmp_var = create_register(); emit(opCONDITION_NE,    $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression CONDITION_EQ expression	{ vm_tmp_var = create_register(); emit(opCONDITION_EQ,    $1, $3, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| expression							{ vm_tmp_var = create_register(); emit(opCONDITION_TRUE,  $1,  0, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
|										{ }
;

/* ═════════════════════════════════════════════════════════════════════════
 *  Expressions (arithmetic, boolean, literals, function calls)
 * ═════════════════════════════════════════════════════════════════════════
 */
expression
: '!' expression						{ vm_tmp_var = create_register(); emit(opNOT, $2,  0, vm_tmp_var); trait_type_set(vm_tmp_var, "boolean", false); $$ = vm_tmp_var; }
| BOOLEAN								{ track($1); trait_type_set($1, "boolean", true); $$ = $1; }
| STRING						        { track($1); trait_type_set($1, "string", true);  $$ = $1; }
| NAME '.' NAME					        { vm_tmp_var = create_register(); emit(opGET_OBJ_VAR, $1, $3, vm_tmp_var); track($1); trait_type_link($3, vm_tmp_var); $$ = vm_tmp_var; }
| math_expression						{ $$ = $1; }
| function_call					        { vm_tmp_var = create_register(); emit(opSTACK_POP, 0,  0, vm_tmp_var); trait_type_set(vm_tmp_var, function_return_type_get($1), false); $$ = vm_tmp_var; }
| json_expression				        { vm_tmp_var = create_register(); $$ = vm_tmp_var; }
| xpath_expression				        { vm_tmp_var = create_register(); $$ = vm_tmp_var; }
;

math_expression
: NUMBER						        { track($1); trait_type_set($1, "number", true);  $$ = $1; }
| NAME						      		{ track($1);  $$ = $1;  }
| '(' math_expression ')'			    { $$ = $2; }
| expression '+' expression		        { vm_tmp_var = create_register(); emit(opADD, $1, $3, vm_tmp_var); trait_type_link($1, vm_tmp_var); $$ = vm_tmp_var; }
| expression '-' expression		        { vm_tmp_var = create_register(); emit(opSUB, $1, $3, vm_tmp_var); trait_type_link($1, vm_tmp_var); $$ = vm_tmp_var; }
| expression '*' expression		        { vm_tmp_var = create_register(); emit(opMUL, $1, $3, vm_tmp_var); trait_type_link($1, vm_tmp_var); $$ = vm_tmp_var; }
| expression '/' expression		        { vm_tmp_var = create_register(); emit(opDIV, $1, $3, vm_tmp_var); trait_type_link($1, vm_tmp_var); $$ = vm_tmp_var; }
| expression '^' expression		        { vm_tmp_var = create_register(); emit(opPOW, $1, $3, vm_tmp_var); trait_type_link($1, vm_tmp_var); $$ = vm_tmp_var; }
| '-' expression %prec UMINUS	        { vm_tmp_var = create_register(); emit(opNEG, $2,  0, vm_tmp_var); trait_type_link($2, vm_tmp_var); $$ = vm_tmp_var; }
;

/* ═════════════════════════════════════════════════════════════════════════
 *  Function calls — parameters pushed onto stack, then CALL_METHOD
 * ═════════════════════════════════════════════════════════════════════════
 */

/* Function calls involve the parameters to be placed onto the stack including
 * a tmp variable for any results
 */
function_call
: NAME '(' expression_params ')'			{ emit(opCALL_METHOD, $1, 0, 0); trait_type_link($1, function_return_type_get($1)); track($1); $$ = $1; }
| NAME '.' NAME '(' expression_params ')'	{ emit(opCALL_OBJ_METHOD, $1, $3, 0); trait_type_link($3, function_return_type_get($3)); track($1); track($3); $$ = $3; /* Needs fix for $1.$3*/ }
;

/* These are the function's parameters which are placed on the stack which
 * will be poped by the function.
 */
expression_params
: expression ',' expression_params	    { emit(opSTACK_PUSH, $1, 0, 0); }
| expression						    { emit(opSTACK_PUSH, $1, 0, 0); }
|									    {}
;

/* ═════════════════════════════════════════════════════════════════════════
 *  LaTeX math — f(x) = expr, summations (\sum_), fractions (\frac)
 * ═════════════════════════════════════════════════════════════════════════
 */

latex_math_function
: comments latex_math_function_header '=' latex_math_expression 	{  	emit(opSTACK_PUSH, $4, 0, 0); emit(opRETURN, 0, 0, 0); 
																		function_end(); 
																	}
;

latex_math_function_header
: latex_math_function_name '(' latex_math_function_parameters ')'	{ function_return_type_set($1, track(string_clone("number"))); $$ = $1; }
;

latex_math_function_parameters
: latex_math_function_parameters ',' latex_math_function_parameter	
| latex_math_function_parameter							
|												
;

latex_math_function_parameter
: NAME 		{ function_parameter($1, "number"); emit(opSTACK_POP, 0, 0, $1); trait_type_set($1, "number", false); track($1); $$ = $1; }
;

latex_math_function_name
: NAME 		{ reset_registers(); function_begin($1); $$ = $1; }
;

latex_math_expression
: opsum_expression			 							{ $$ = $1; }
| OPERATOR_FRAC '{' expression '}' '{' expression '}' 	{ $$ = $3; }
//| OPERATOR_POW ''
| math_expression										{ $$ = $1; }
;

opsum_expression
: opsum latex_math_expression	{ 
		comment("sum-of: end");
		comment("sum-of: summation");
		emit(opADD, $1, $2, $1); trait_type_link($1, $2); 

		/* We should have our summation iterator stashed away */
		emit_defer_pop();

		/* Perform the summation loop */
		vm_tmp_label = labels_pop(); emit(opJUMP, 0, 0, labels_pop()); emit(opLABEL, vm_tmp_label, 0, 0); 

		/* Return the variable with the summation */
		$$ = $1;
	}
;

opsum
: OPERATOR_SUM '{' NAME '=' math_expression '}' '^' math_expression			
	{ 
		comment("sum-of: begin");

		/* Create the initial start of the summation loop */
		emit(opCREATE_SET_VAR, $5, 0, $3); trait_type_link($5, $3); track($3); 	

		/* When we execute the expression we want to sum it */
		vm_tmp_var = create_register();
		trait_type_link($3, vm_tmp_var); 

		/* Return the variable that has summation value */
		$$ = vm_tmp_var; 

		/* The label where we will loop back to */
		vm_tmp_label = create_label(); emit(opLABEL, vm_tmp_label, 0, 0); labels_push(vm_tmp_label); 
															
		/* Has the summation limit been reached? */
		vm_tmp_var = create_register(); emit(opCONDITION_LT, $3, $8, vm_tmp_var);
		trait_type_set($8, "number", false); 
		trait_type_set(vm_tmp_var, "boolean", false);

		/* If yes, the exit the loop */
		vm_tmp_label = create_label(); 
		emit(opJUMP_IF_FALSE, vm_tmp_var, 0, vm_tmp_label); 
		labels_push(vm_tmp_label);
		comment("apply-to: begin");

		/* We want to perform increment after the summation expression is completed, so defer these instructions */
		emit_defer_push_begin(); 
		comment("apply-to: end");
		comment("sum-of: increment");
		emit(opINC, $3, 0, 0); 
		emit_defer_push_end();
	}
;

/* ═════════════════════════════════════════════════════════════════════════
 *  XPath expressions (legacy / experimental)
 * ═════════════════════════════════════════════════════════════════════════
 */

xpath_expression
: xpath_fragments
;

xpath_fragments
: xpath_fragment xpath_fragments
| xpath_fragment
|
;

xpath_fragment
: '/' NAME
| '/' NAME '[' conditional_expressions ']'

/* ═════════════════════════════════════════════════════════════════════════
 *  JSON expressions (legacy / experimental)
 * ═════════════════════════════════════════════════════════════════════════
 */
json_expression
: '[' json_array_items ']'
| '{' json_tuples '}'
;

json_array_items
: json_array_items ',' json_value
| json_value
|
;

json_tuples
: json_tuples ',' json_tuple
| json_tuple
|
;

json_tuple
: STRING ':' json_value
| NAME ':' json_value
;

json_value
: NUMBER
| STRING
| json_expression
;

%%