/*
 * microgpt_vm_parser.tab.c  —  Auto-generated Bison parser output.
 *
 * Generated from microgpt_vm.y — do not edit directly.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

/* A Bison parser, made by GNU Bison 2.7.  */

/* Bison implementation for Yacc-like parsers in C

      Copyright (C) 1984, 1989-1990, 2000-2012 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.7"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE VM_MODULE_PARSER_STYPE
/* Substitute the variable and function names.  */
#define yyparse vm_module_parser_parse
#define yylex vm_module_parser_lex
#define yyerror vm_module_parser_error
#define yylval vm_module_parser_lval
#define yychar vm_module_parser_char
#define yydebug vm_module_parser_debug
#define yynerrs vm_module_parser_nerrs

/* Copy the first part of user declarations.  */
/* Line 371 of yacc.c  */
#line 10 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"

#include "microgpt_vm.h"

#include <stdio.h>

extern void vm_module_parser_error(vm_module_parser *parser, char *error);
extern int yylex(vm_module_parser *parser);

char *vm_tmp_var = 0;
char *vm_tmp_label = 0;
char *vm_tmp_type = 0;

queue *tracking_labels;

#define function_begin(name)                                                   \
  vm_module_generator_function_begin(parser->generator, name)
#define function_end() vm_module_generator_function_end(parser->generator)
#define function_parameter(name, type)                                         \
  vm_module_generator_function_parameter(parser->generator, name, type)
#define function_return_type_set(name, type)                                   \
  vm_module_generator_function_return_type_set(parser->generator, name,        \
                                               (char *)type)
#define function_return_type_get(name)                                         \
  vm_module_generator_function_return_type_get(parser->generator, name)
#define emit(opcode, param1, param2, param3)                                   \
  vm_module_generator_function_emit_with_meta(                                 \
      parser->generator, opcode, (char *)param1, (char *)param2,               \
      (char *)param3, (char *)parser->vm_module_parser_state_input,            \
      parser->vm_module_parser_state_input_line_number,                        \
      parser->vm_module_parser_state_input_index)
#define emit_defer_push_begin()                                                \
  vm_module_generator_defer_push_begin(parser->generator)
#define emit_defer_push_end()                                                  \
  vm_module_generator_defer_push_end(parser->generator)
#define emit_defer_pop() vm_module_generator_defer_pop(parser->generator)
#define create_register()                                                      \
  vm_module_generator_tmp_register_create(parser->generator)
#define reset_registers()                                                      \
  vm_module_generator_tmp_registers_reset(parser->generator)
#define create_label() vm_module_generator_tmp_label_create(parser->generator)
#define labels_pop() vm_module_generator_tracking_labels_pop(parser->generator)
#define labels_push(symbol)                                                    \
  vm_module_generator_tracking_labels_push(parser->generator, symbol)
#define track(symbol)                                                          \
  vm_module_generator_symbol_track(parser->generator, symbol)
#define trait_type_set(symbol, type, is_constant)                              \
  vm_module_generator_trait_type_set(parser->generator, symbol, type,          \
                                     is_constant)
#define trait_type_get(symbol)                                                 \
  vm_module_generator_trait_type_get(parser->generator, symbol)
#define trait_type_link(source, target)                                        \
  trait_type_set(target, trait_type_get(source), false);                       \
  trait_type_set(source, trait_type_get(target), false);
#define comment(message)                                                       \
  vm_module_generator_function_emit_comment_with_meta(                         \
      parser->generator, (char *)message,                                      \
      (char *)parser->vm_module_parser_state_input,                            \
      parser->vm_module_parser_state_input_line_number,                        \
      parser->vm_module_parser_state_input_index)

/* Line 371 of yacc.c  */
#line 113 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.tab.cpp"

#ifndef YY_NULL
#if defined __cplusplus && 201103L <= __cplusplus
#define YY_NULL nullptr
#else
#define YY_NULL 0
#endif
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
#undef YYERROR_VERBOSE
#define YYERROR_VERBOSE 1
#else
#define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "vm_module_parser.tab.hpp".  */
#ifndef YY_VM_MODULE_PARSER_D_DEV_PROJECTS_ENJECTOR_ENJECTOR_VM_MAIN_CORE_SRC_VM_MODULE_PARSER_TAB_HPP_INCLUDED
#define YY_VM_MODULE_PARSER_D_DEV_PROJECTS_ENJECTOR_ENJECTOR_VM_MAIN_CORE_SRC_VM_MODULE_PARSER_TAB_HPP_INCLUDED
/* Enabling traces.  */
#ifndef VM_MODULE_PARSER_DEBUG
#if defined YYDEBUG
#if YYDEBUG
#define VM_MODULE_PARSER_DEBUG 1
#else
#define VM_MODULE_PARSER_DEBUG 0
#endif
#else /* ! defined YYDEBUG */
#define VM_MODULE_PARSER_DEBUG 0
#endif /* ! defined YYDEBUG */
#endif /* ! defined VM_MODULE_PARSER_DEBUG */
#if VM_MODULE_PARSER_DEBUG
extern int vm_module_parser_debug;
#endif

/* Tokens.  */
#ifndef VM_MODULE_PARSER_TOKENTYPE
#define VM_MODULE_PARSER_TOKENTYPE
/* Put the tokens into the symbol table, so that GDB and other debuggers
   know about them.  */
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

#if !defined VM_MODULE_PARSER_STYPE &&                                         \
    !defined VM_MODULE_PARSER_STYPE_IS_DECLARED
typedef union VM_MODULE_PARSER_STYPE {
/* Line 387 of yacc.c  */
#line 55 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"

  char *string;

/* Line 387 of yacc.c  */
#line 201 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.tab.cpp"
} VM_MODULE_PARSER_STYPE;
#define VM_MODULE_PARSER_STYPE_IS_TRIVIAL 1
#define vm_module_parser_stype                                                 \
  VM_MODULE_PARSER_STYPE /* obsolescent; will be withdrawn */
#define VM_MODULE_PARSER_STYPE_IS_DECLARED 1
#endif

extern VM_MODULE_PARSER_STYPE vm_module_parser_lval;

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int vm_module_parser_parse(void *YYPARSE_PARAM);
#else
int vm_module_parser_parse();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int vm_module_parser_parse(vm_module_parser *parser);
#else
int vm_module_parser_parse();
#endif
#endif /* ! YYPARSE_PARAM */

#endif /* !YY_VM_MODULE_PARSER_D_DEV_PROJECTS_ENJECTOR_ENJECTOR_VM_MAIN_CORE_SRC_VM_MODULE_PARSER_TAB_HPP_INCLUDED \
        */

/* Copy the second part of user declarations.  */

/* Line 390 of yacc.c  */
#line 229 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.tab.cpp"

#ifdef short
#undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||     \
       defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
#ifdef __SIZE_TYPE__
#define YYSIZE_T __SIZE_TYPE__
#elif defined size_t
#define YYSIZE_T size_t
#elif !defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ ||       \
                            defined __cplusplus || defined _MSC_VER)
#include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#define YYSIZE_T size_t
#else
#define YYSIZE_T unsigned int
#endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) - 1)

#ifndef YY_
#if defined YYENABLE_NLS && YYENABLE_NLS
#if ENABLE_NLS
#include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#define YY_(Msgid) dgettext("bison-runtime", Msgid)
#endif
#endif
#ifndef YY_
#define YY_(Msgid) Msgid
#endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if !defined lint || defined __GNUC__
#define YYUSE(E) ((void)(E))
#else
#define YYUSE(E) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
#define YYID(N) (N)
#else
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static int YYID(int yyi)
#else
static int YYID(yyi)
int yyi;
#endif
{
  return yyi;
}
#endif

#if !defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

#ifdef YYSTACK_USE_ALLOCA
#if YYSTACK_USE_ALLOCA
#ifdef __GNUC__
#define YYSTACK_ALLOC __builtin_alloca
#elif defined __BUILTIN_VA_ARG_INCR
#include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#elif defined _AIX
#define YYSTACK_ALLOC __alloca
#elif defined _MSC_VER
#include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#define alloca _alloca
#else
#define YYSTACK_ALLOC alloca
#if !defined _ALLOCA_H && !defined EXIT_SUCCESS &&                             \
    (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
/* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#endif
#endif
#endif

#ifdef YYSTACK_ALLOC
/* Pacify GCC's `empty if-body' warning.  */
#define YYSTACK_FREE(Ptr)                                                      \
  do { /* empty */                                                             \
    ;                                                                          \
  } while (YYID(0))
#ifndef YYSTACK_ALLOC_MAXIMUM
/* The OS might guarantee only one guard page at the bottom of the stack,
   and a page size can be as small as 4096 bytes.  So we cannot safely
   invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
   to allow for a few compiler-allocated temporary stack slots.  */
#define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#endif
#else
#define YYSTACK_ALLOC YYMALLOC
#define YYSTACK_FREE YYFREE
#ifndef YYSTACK_ALLOC_MAXIMUM
#define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#endif
#if (defined __cplusplus && !defined EXIT_SUCCESS &&                           \
     !((defined YYMALLOC || defined malloc) &&                                 \
       (defined YYFREE || defined free)))
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#ifndef YYMALLOC
#define YYMALLOC malloc
#if !defined malloc && !defined EXIT_SUCCESS &&                                \
    (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
void *malloc(YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#ifndef YYFREE
#define YYFREE free
#if !defined free && !defined EXIT_SUCCESS &&                                  \
    (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
void free(void *); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */

#if (!defined yyoverflow &&                                                    \
     (!defined __cplusplus || (defined VM_MODULE_PARSER_STYPE_IS_TRIVIAL &&    \
                               VM_MODULE_PARSER_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc {
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
#define YYSTACK_GAP_MAXIMUM (sizeof(union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
#define YYSTACK_BYTES(N)                                                       \
  ((N) * (sizeof(yytype_int16) + sizeof(YYSTYPE)) + YYSTACK_GAP_MAXIMUM)

#define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
#define YYSTACK_RELOCATE(Stack_alloc, Stack)                                   \
  do {                                                                         \
    YYSIZE_T yynewbytes;                                                       \
    YYCOPY(&yyptr->Stack_alloc, Stack, yysize);                                \
    Stack = &yyptr->Stack_alloc;                                               \
    yynewbytes = yystacksize * sizeof(*Stack) + YYSTACK_GAP_MAXIMUM;           \
    yyptr += yynewbytes / sizeof(*yyptr);                                      \
  } while (YYID(0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
#ifndef YYCOPY
#if defined __GNUC__ && 1 < __GNUC__
#define YYCOPY(Dst, Src, Count)                                                \
  __builtin_memcpy(Dst, Src, (Count) * sizeof(*(Src)))
#else
#define YYCOPY(Dst, Src, Count)                                                \
  do {                                                                         \
    YYSIZE_T yyi;                                                              \
    for (yyi = 0; yyi < (Count); yyi++)                                        \
      (Dst)[yyi] = (Src)[yyi];                                                 \
  } while (YYID(0))
#endif
#endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL 3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST 478

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS 50
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS 47
/* YYNRULES -- Number of rules.  */
#define YYNRULES 123
/* YYNRULES -- Number of states.  */
#define YYNSTATES 245

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK 2
#define YYMAXUTOK 287

#define YYTRANSLATE(YYX)                                                       \
  ((unsigned int)(YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] = {
    0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  47, 2,  2,
    2,  2,  2,  2,  41, 42, 34, 33, 44, 32, 46, 35, 2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  43, 40, 2,  45, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  48, 2,  49, 36, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  38, 2,  39,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37};

#if VM_MODULE_PARSER_DEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] = {
    0,   0,   3,   7,   10,  11,  13,  16,  17,  23,  26,  29,  31,  38,
    43,  51,  57,  60,  64,  66,  67,  71,  75,  77,  78,  82,  85,  86,
    89,  94,  99,  106, 113, 116, 119, 123, 127, 130, 136, 141, 146, 151,
    156, 159, 162, 164, 167, 170, 174, 178, 183, 188, 193, 199, 204, 206,
    212, 217, 219, 221, 225, 229, 233, 235, 239, 243, 247, 251, 255, 259,
    261, 262, 265, 267, 269, 273, 275, 277, 279, 281, 283, 285, 289, 293,
    297, 301, 305, 309, 312, 317, 324, 328, 330, 331, 336, 341, 345, 347,
    348, 350, 352, 354, 362, 364, 367, 376, 378, 381, 383, 384, 387, 393,
    397, 401, 405, 407, 408, 412, 414, 415, 419, 423, 425, 427};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] = {
    51, 0,  -1, 52, 54, 52, -1, 52, 53, -1, -1, 7,  -1, 54, 55, -1, -1, 52, 56,
    38, 63, 39, -1, 57, 40, -1, 81, 40, -1, 81, -1, 58, 41, 59, 42, 43, 3,  -1,
    58, 41, 59, 42, -1, 9,  58, 41, 61, 42, 43, 3,  -1, 9,  58, 41, 61, 42, -1,
    10, 3,  -1, 59, 44, 60, -1, 60, -1, -1, 3,  43, 3,  -1, 61, 44, 62, -1, 62,
    -1, -1, 3,  43, 3,  -1, 63, 64, -1, -1, 66, 40, -1, 3,  45, 77, 40, -1, 3,
    45, 75, 40, -1, 3,  46, 3,  45, 77, 40, -1, 3,  46, 3,  45, 75, 40, -1, 65,
    40, -1, 79, 40, -1, 11, 77, 40, -1, 11, 75, 40, -1, 11, 40, -1, 12, 41, 3,
    42, 40, -1, 69, 38, 63, 39, -1, 71, 38, 63, 39, -1, 67, 38, 63, 39, -1, 68,
    38, 63, 39, -1, 68, 64, -1, 67, 64, -1, 7,  -1, 3,  25, -1, 3,  26, -1, 3,
    27, 77, -1, 3,  28, 77, -1, 8,  3,  45, 77, -1, 8,  3,  45, 75, -1, 13, 41,
    75, 42, -1, 67, 38, 63, 39, 14, -1, 70, 41, 75, 42, -1, 23, -1, 72, 73, 40,
    74, 42, -1, 24, 41, 66, 40, -1, 75, -1, 65, -1, 41, 75, 42, -1, 75, 21, 76,
    -1, 75, 22, 76, -1, 76, -1, 77, 15, 77, -1, 77, 16, 77, -1, 77, 17, 77, -1,
    77, 18, 77, -1, 77, 19, 77, -1, 77, 20, 77, -1, 77, -1, -1, 47, 77, -1, 6,
    -1, 5,  -1, 3,  46, 3,  -1, 78, -1, 79, -1, 92, -1, 89, -1, 4,  -1, 3,  -1,
    41, 78, 42, -1, 77, 33, 77, -1, 77, 32, 77, -1, 77, 34, 77, -1, 77, 35, 77,
    -1, 77, 36, 77, -1, 32, 77, -1, 3,  41, 80, 42, -1, 3,  46, 3,  41, 80, 42,
    -1, 77, 44, 80, -1, 77, -1, -1, 52, 82, 45, 86, -1, 85, 41, 83, 42, -1, 83,
    44, 84, -1, 84, -1, -1, 3,  -1, 3,  -1, 87, -1, 30, 38, 77, 39, 38, 77, 39,
    -1, 78, -1, 88, 86, -1, 29, 38, 3,  45, 78, 39, 36, 78, -1, 90, -1, 91, 90,
    -1, 91, -1, -1, 35, 3,  -1, 35, 3,  48, 75, 49, -1, 48, 93, 49, -1, 38, 94,
    39, -1, 93, 44, 96, -1, 96, -1, -1, 94, 44, 95, -1, 95, -1, -1, 5,  43, 96,
    -1, 3,  43, 96, -1, 4,  -1, 5,  -1, 92, -1};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] = {
    0,   104, 104, 108, 109, 113, 119, 121, 127, 128, 129, 130, 134, 135,
    139, 140, 146, 153, 154, 155, 159, 163, 164, 165, 169, 176, 177, 186,
    187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 201, 208, 209,
    210, 211, 212, 216, 217, 218, 219, 223, 224, 228, 232, 236, 240, 246,
    250, 256, 263, 267, 268, 269, 270, 274, 275, 276, 277, 278, 279, 280,
    281, 288, 289, 290, 291, 292, 293, 294, 295, 299, 300, 301, 302, 303,
    304, 305, 306, 307, 318, 319, 326, 327, 328, 336, 342, 346, 347, 348,
    352, 356, 360, 361, 363, 367, 384, 426, 430, 431, 432, 436, 437, 443,
    444, 448, 449, 450, 454, 455, 456, 460, 461, 465, 466, 467};
#endif

#if VM_MODULE_PARSER_DEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] = {"$end",
                                      "error",
                                      "$undefined",
                                      "NAME",
                                      "NUMBER",
                                      "STRING",
                                      "BOOLEAN",
                                      "COMMENT",
                                      "VAR",
                                      "DECLARE",
                                      "FUNCTION",
                                      "RETURN",
                                      "YIELD",
                                      "IF",
                                      "ELSE",
                                      "CONDITION_GTE",
                                      "CONDITION_LTE",
                                      "CONDITION_GT",
                                      "CONDITION_LT",
                                      "CONDITION_NE",
                                      "CONDITION_EQ",
                                      "CONDITIONAL_AND",
                                      "CONDITIONAL_OR",
                                      "WHILE",
                                      "FOR",
                                      "INCREMENT_BY_ONE",
                                      "DECREMENT_BY_ONE",
                                      "INCREMENT_BY",
                                      "DECREMENT_BY",
                                      "OPERATOR_SUM",
                                      "OPERATOR_FRAC",
                                      "OPERATOR_POW",
                                      "'-'",
                                      "'+'",
                                      "'*'",
                                      "'/'",
                                      "'^'",
                                      "UMINUS",
                                      "'{'",
                                      "'}'",
                                      "';'",
                                      "'('",
                                      "')'",
                                      "':'",
                                      "','",
                                      "'='",
                                      "'.'",
                                      "'!'",
                                      "'['",
                                      "']'",
                                      "$accept",
                                      "code",
                                      "comments",
                                      "comment",
                                      "functions",
                                      "function",
                                      "function_header",
                                      "declare_function_header",
                                      "function_name",
                                      "function_parameters",
                                      "function_parameter",
                                      "declare_function_parameters",
                                      "declare_function_parameter",
                                      "statements",
                                      "statement",
                                      "increment_expression",
                                      "var_declaration",
                                      "if_condition",
                                      "if_condition_else",
                                      "while_statement",
                                      "while",
                                      "for_statement",
                                      "for",
                                      "for_conditional_loop",
                                      "for_increment_expression",
                                      "conditional_expressions",
                                      "conditional_expression",
                                      "expression",
                                      "math_expression",
                                      "function_call",
                                      "expression_params",
                                      "latex_math_function",
                                      "latex_math_function_header",
                                      "latex_math_function_parameters",
                                      "latex_math_function_parameter",
                                      "latex_math_function_name",
                                      "latex_math_expression",
                                      "opsum_expression",
                                      "opsum",
                                      "xpath_expression",
                                      "xpath_fragments",
                                      "xpath_fragment",
                                      "json_expression",
                                      "json_array_items",
                                      "json_tuples",
                                      "json_tuple",
                                      "json_value",
                                      YY_NULL};
#endif

#ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] = {
    0,   256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
    281, 282, 283, 284, 285, 286, 45,  43,  42,  47,  94,  287, 123,
    125, 59,  40,  41,  58,  44,  61,  46,  33,  91,  93};
#endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] = {
    0,  50, 51, 52, 52, 53, 54, 54, 55, 55, 55, 55, 56, 56, 57, 57, 58, 59,
    59, 59, 60, 61, 61, 61, 62, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 77,
    77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79,
    80, 80, 80, 81, 82, 83, 83, 83, 84, 85, 86, 86, 86, 87, 88, 89, 90, 90,
    90, 91, 91, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 96, 96, 96};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] = {
    0, 2, 3, 2, 0, 1, 2, 0, 5, 2, 2, 1, 6, 4, 7, 5, 2, 3, 1, 0, 3, 3, 1, 0, 3,
    2, 0, 2, 4, 4, 6, 6, 2, 2, 3, 3, 2, 5, 4, 4, 4, 4, 2, 2, 1, 2, 2, 3, 3, 4,
    4, 4, 5, 4, 1, 5, 4, 1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 0, 2, 1, 1, 3,
    1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 4, 6, 3, 1, 0, 4, 4, 3, 1, 0, 1, 1,
    1, 7, 1, 2, 8, 1, 2, 1, 0, 2, 5, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 1, 1, 1};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] = {
    4,   0,   7,   1,   5,   3,   4,   0,   2,   6,   0,   11,  0,   0,   99,
    0,   0,   0,   0,   9,   10,  16,  23,  26,  19,  108, 97,  0,   0,   22,
    0,   0,   0,   18,  80,  79,  73,  72,  0,   0,   108, 0,   118, 108, 108,
    115, 0,   102, 76,  93,  100, 108, 78,  105, 107, 77,  98,  0,   96,  0,
    15,  0,   0,   44,  0,   108, 0,   0,   54,  0,   8,   25,  0,   0,   0,
    0,   0,   0,   0,   108, 0,   0,   13,  0,   108, 0,   0,   108, 87,  75,
    109, 0,   0,   0,   117, 75,  71,  121, 122, 123, 0,   114, 108, 108, 108,
    108, 108, 103, 106, 94,  0,   24,  0,   21,  45,  46,  108, 108, 108, 0,
    0,   36,  108, 0,   62,  69,  0,   108, 0,   32,  27,  26,  43,  26,  42,
    26,  108, 26,  0,   57,  69,  33,  20,  0,   17,  91,  0,   74,  0,   0,
    108, 0,   0,   112, 0,   81,  0,   111, 83,  82,  84,  85,  86,  95,  14,
    47,  48,  0,   69,  0,   108, 0,   108, 108, 35,  108, 108, 108, 108, 108,
    108, 34,  0,   0,   0,   0,   0,   0,   0,   0,   0,   12,  108, 88,  108,
    108, 0,   0,   120, 119, 116, 113, 29,  28,  108, 50,  69,  59,  60,  61,
    63,  64,  65,  66,  67,  68,  0,   51,  56,  40,  41,  38,  53,  39,  0,
    58,  0,   90,  0,   75,  108, 110, 0,   69,  37,  52,  55,  89,  0,   0,
    31,  30,  108, 101, 104};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] = {
    -1, 1,  2,  5,  6,  9,  15, 10,  13,  32,  33,  28,  29, 30, 71,  72,
    73, 74, 75, 76, 77, 78, 79, 138, 226, 123, 124, 140, 89, 48, 146, 11,
    17, 57, 58, 18, 49, 50, 51, 52,  53,  54,  55,  100, 93, 94, 101};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -76
static const yytype_int16 yypact[] = {
    -76, 25,  29,  -76, -76, -76, 40,  63,  53,  -76, 58,  68,  108, 75,  -76,
    89,  103, 118, 125, -76, -76, -76, 156, -76, 165, 3,   168, 130, 22,  -76,
    287, 131, 52,  -76, -36, -76, -76, -76, 138, 139, 245, 177, 114, 245, 245,
    17,  437, 442, -76, -76, -76, 3,   -76, -76, 147, -76, -76, 78,  -76, 180,
    142, 156, 161, -76, 198, 190, 162, 163, -76, 167, -76, -76, 169, 170, 338,
    361, 175, 173, 182, 194, 178, 218, 181, 165, 65,  224, 230, 245, -76, -76,
    188, 196, 197, 28,  -76, 205, 437, -76, -76, -76, 41,  -76, 245, 245, 245,
    245, 245, -76, -76, -76, 168, -76, 260, -76, -76, -76, 245, 245, 194, 262,
    226, -76, 240, 2,   -76, 362, 271, 240, 215, -76, -76, -76, -76, -76, -76,
    -76, 240, -76, 236, 80,  432, -76, -76, 276, -76, 220, 242, 244, 246, 101,
    143, 17,  17,  -76, 114, -76, 17,  -76, 48,  48,  253, 253, -76, -76, -76,
    437, 437, 8,   371, -14, 194, -8,  83,  83,  -76, 245, 245, 245, 245, 245,
    245, -76, 254, -5,  263, 294, 301, 324, 32,  331, 312, -76, 65,  -76, 65,
    245, 278, -10, -76, -76, -76, -76, -76, -76, 194, 80,  397, -76, -76, -76,
    437, 437, 437, 437, 437, 437, 279, -76, -76, 306, -76, -76, -76, -76, 232,
    -76, 280, -76, 281, 282, 245, -76, 36,  406, -76, -76, -76, -76, 292, 234,
    -76, -76, 245, -76, 442};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] = {
    -76, -76, 323, -76, -76, -76, -76, -76, 344, -76, 247, -76,
    295, 322, 67,  176, 225, -76, -76, -76, -76, -76, -76, -76,
    -76, -75, -11, -25, -23, -29, -66, -76, -76, -76, 248, -76,
    308, -76, -76, -76, 303, -76, -42, -76, -76, 206, -13};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -93
static const yytype_int16 yytable[] = {
    46,  80,  47,  99,  139, 84,  34,  35,  36,  37,  85,  172, 173, 172, 173,
    88,  172, 173, 46,  96,  95,  97,  98,  172, 173, 3,   46,  194, 47,  172,
    173, 204, 38,  39,  207, 40,  4,   217, 41,  231, 125, 42,  174, 167, 43,
    80,  80,  171, 202, 7,   44,  45,  183, 172, 173, 42,  14,  172, 173, 145,
    4,   188, 149, 12,  60,  45,  61,  153, 34,  35,  36,  37,  154, 12,  222,
    197, 240, 158, 159, 160, 161, 162, 104, 105, 106, 156, 34,  35,  36,  37,
    157, 165, 166, 168, 82,  205, 83,  40,  19,  95,  41,  172, 173, 42,  -70,
    -70, 43,  -92, 20,  99,  99,  21,  44,  45,  99,  40,  22,  91,  41,  92,
    109, 42,  110, -70, 43,  -70, 227, 23,  228, 232, 44,  45,  -70, 102, 103,
    104, 105, 106, 198, 199, 196, 132, 134, 201, 24,  206, 34,  35,  36,  37,
    210, 211, 212, 213, 214, 215, 80,  80,  80,  27,  80,  208, 209, 25,  -70,
    -70, 26,  145, 31,  145, 46,  56,  229, 59,  81,  40,  86,  87,  41,  233,
    90,  42,  41,  111, 122, 112, 114, 115, 116, 117, 44,  45,  -70, 34,  35,
    36,  37,  34,  35,  36,  37,  120, 84,  126, 127, 239, 118, 119, 128, 129,
    130, -70, -70, 135, 136, -70, -70, 46,  141, 244, 137, 142, 40,  64,  143,
    41,  40,  147, 42,  41,  121, 122, 42,  148, -70, 122, 150, 44,  45,  151,
    152, 44,  45,  34,  35,  36,  37,  155, 34,  35,  36,  37,  102, 103, 104,
    105, 106, 114, 115, 116, 117, -70, -70, 164, 192, 169, 102, 103, 104, 105,
    106, 170, 40,  243, 182, 41,  190, 40,  42,  191, 41,  122, -70, 42,  193,
    194, 43,  44,  45,  106, 62,  195, 44,  45,  63,  64,  216, 62,  65,  66,
    67,  63,  64,  218, 62,  65,  66,  67,  63,  64,  68,  69,  65,  66,  67,
    224, 230, 68,  69,  234, 235, 238, 236, 237, 68,  69,  70,  62,  242, 8,
    144, 63,  64,  219, 62,  65,  66,  67,  63,  64,  220, 62,  65,  66,  67,
    63,  64,  68,  69,  65,  66,  67,  16,  184, 68,  69,  113, 108, 163, 107,
    200, 68,  69,  221, 62,  0,   225, 0,   63,  64,  223, 0,   65,  66,  67,
    0,   131, 175, 176, 177, 178, 179, 180, 0,   68,  69,  175, 176, 177, 178,
    179, 180, 0,   0,   102, 103, 104, 105, 106, 133, 0,   0,   181, 102, 103,
    104, 105, 106, 0,   0,   0,   203, 175, 176, 177, 178, 179, 180, 0,   0,
    0,   175, 176, 177, 178, 179, 180, 0,   0,   102, 103, 104, 105, 106, 0,
    0,   0,   -49, 102, 103, 104, 105, 106, 0,   0,   0,   241, 175, 176, 177,
    178, 179, 180, 185, 0,   186, 0,   187, 0,   189, 0,   0,   0,   0,   102,
    103, 104, 105, 106, 102, 103, 104, 105, 106, -75, -75, -75, -75, -75};

#define yypact_value_is_default(Yystate) (!!((Yystate) == (-76)))

#define yytable_value_is_error(Yytable_value) YYID(0)

static const yytype_int16 yycheck[] = {
    25,  30,  25,  45,  79,  41,  3,   4,   5,   6,   46,  21,  22,  21,  22,
    40,  21,  22,  43,  44,  43,  4,   5,   21,  22,  0,   51,  41,  51,  21,
    22,  45,  29,  30,  42,  32,  7,   42,  35,  49,  65,  38,  40,  118, 41,
    74,  75,  122, 40,  9,   47,  48,  127, 21,  22,  38,  3,   21,  22,  84,
    7,   136, 87,  10,  42,  48,  44,  39,  3,   4,   5,   6,   44,  10,  42,
    150, 40,  102, 103, 104, 105, 106, 34,  35,  36,  44,  3,   4,   5,   6,
    49,  116, 117, 118, 42,  170, 44,  32,  40,  122, 35,  21,  22,  38,  21,
    22,  41,  42,  40,  151, 152, 3,   47,  48,  156, 32,  41,  3,   35,  5,
    42,  38,  44,  40,  41,  42,  192, 38,  194, 204, 47,  48,  49,  32,  33,
    34,  35,  36,  151, 152, 39,  74,  75,  156, 41,  170, 3,   4,   5,   6,
    175, 176, 177, 178, 179, 180, 185, 186, 187, 3,   189, 172, 173, 45,  21,
    22,  41,  192, 3,   194, 195, 3,   195, 43,  43,  32,  38,  38,  35,  204,
    3,   38,  35,  3,   41,  43,  25,  26,  27,  28,  47,  48,  49,  3,   4,
    5,   6,   3,   4,   5,   6,   3,   41,  41,  41,  230, 45,  46,  41,  40,
    40,  21,  22,  38,  41,  21,  22,  242, 40,  242, 38,  3,   32,  8,   43,
    35,  32,  3,   38,  35,  40,  41,  38,  3,   40,  41,  48,  47,  48,  43,
    43,  47,  48,  3,   4,   5,   6,   42,  3,   4,   5,   6,   32,  33,  34,
    35,  36,  25,  26,  27,  28,  21,  22,  3,   44,  3,   32,  33,  34,  35,
    36,  45,  32,  39,  3,   35,  40,  32,  38,  3,   35,  41,  42,  38,  42,
    41,  41,  47,  48,  36,  3,   45,  47,  48,  7,   8,   42,  3,   11,  12,
    13,  7,   8,   40,  3,   11,  12,  13,  7,   8,   23,  24,  11,  12,  13,
    3,   38,  23,  24,  40,  14,  39,  42,  42,  23,  24,  39,  3,   36,  6,
    83,  7,   8,   39,  3,   11,  12,  13,  7,   8,   39,  3,   11,  12,  13,
    7,   8,   23,  24,  11,  12,  13,  8,   128, 23,  24,  61,  54,  110, 51,
    154, 23,  24,  39,  3,   -1,  190, -1,  7,   8,   39,  -1,  11,  12,  13,
    -1,  38,  15,  16,  17,  18,  19,  20,  -1,  23,  24,  15,  16,  17,  18,
    19,  20,  -1,  -1,  32,  33,  34,  35,  36,  38,  -1,  -1,  40,  32,  33,
    34,  35,  36,  -1,  -1,  -1,  40,  15,  16,  17,  18,  19,  20,  -1,  -1,
    -1,  15,  16,  17,  18,  19,  20,  -1,  -1,  32,  33,  34,  35,  36,  -1,
    -1,  -1,  40,  32,  33,  34,  35,  36,  -1,  -1,  -1,  40,  15,  16,  17,
    18,  19,  20,  131, -1,  133, -1,  135, -1,  137, -1,  -1,  -1,  -1,  32,
    33,  34,  35,  36,  32,  33,  34,  35,  36,  32,  33,  34,  35,  36};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] = {
    0,  51, 52, 0,  7,  53, 54, 9,  52, 55, 57, 81, 10, 58, 3,  56, 58, 82, 85,
    40, 40, 3,  41, 38, 41, 45, 41, 3,  61, 62, 63, 3,  59, 60, 3,  4,  5,  6,
    29, 30, 32, 35, 38, 41, 47, 48, 77, 78, 79, 86, 87, 88, 89, 90, 91, 92, 3,
    83, 84, 43, 42, 44, 3,  7,  8,  11, 12, 13, 23, 24, 39, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 79, 43, 42, 44, 41, 46, 38, 38, 77, 78, 3,  3,  5,  94, 95,
    78, 77, 4,  5,  92, 93, 96, 32, 33, 34, 35, 36, 86, 90, 42, 44, 3,  43, 62,
    25, 26, 27, 28, 45, 46, 3,  40, 41, 75, 76, 77, 41, 41, 41, 40, 40, 38, 64,
    38, 64, 38, 41, 38, 73, 75, 77, 40, 3,  43, 60, 77, 80, 3,  3,  77, 48, 43,
    43, 39, 44, 42, 44, 49, 77, 77, 77, 77, 77, 84, 3,  77, 77, 75, 77, 3,  45,
    75, 21, 22, 40, 15, 16, 17, 18, 19, 20, 40, 3,  75, 66, 63, 63, 63, 75, 63,
    40, 3,  44, 42, 41, 45, 39, 75, 96, 96, 95, 96, 40, 40, 45, 75, 77, 42, 76,
    76, 77, 77, 77, 77, 77, 77, 42, 42, 40, 39, 39, 39, 42, 39, 3,  65, 74, 80,
    80, 78, 38, 49, 75, 77, 40, 14, 42, 42, 39, 77, 40, 40, 36, 39, 78};

#define yyerrok (yyerrstatus = 0)
#define yyclearin (yychar = YYEMPTY)
#define YYEMPTY (-2)
#define YYEOF 0

#define YYACCEPT goto yyacceptlab
#define YYABORT goto yyabortlab
#define YYERROR goto yyerrorlab

/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL goto yyerrlab
#if defined YYFAIL
/* This is here to suppress warnings from the GCC cpp's
   -Wunused-macros.  Normally we don't worry about that warning, but
   some users do, and we want to make it easy for users to remove
   YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING() (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                                 \
  do                                                                           \
    if (yychar == YYEMPTY) {                                                   \
      yychar = (Token);                                                        \
      yylval = (Value);                                                        \
      YYPOPSTACK(yylen);                                                       \
      yystate = *yyssp;                                                        \
      goto yybackup;                                                           \
    } else {                                                                   \
      yyerror(parser, YY_("syntax error: cannot back up"));                    \
      YYERROR;                                                                 \
    }                                                                          \
  while (YYID(0))

/* Error token number */
#define YYTERROR 1
#define YYERRCODE 256

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
#define YY_LOCATION_PRINT(File, Loc) ((void)0)
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */
#ifdef YYLEX_PARAM
#define YYLEX yylex(YYLEX_PARAM)
#else
#define YYLEX yylex(parser)
#endif

/* Enable debugging if requested.  */
#if VM_MODULE_PARSER_DEBUG

#ifndef YYFPRINTF
#include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#define YYFPRINTF fprintf
#endif

#define YYDPRINTF(Args)                                                        \
  do {                                                                         \
    if (yydebug)                                                               \
      YYFPRINTF Args;                                                          \
  } while (YYID(0))

#define YY_SYMBOL_PRINT(Title, Type, Value, Location)                          \
  do {                                                                         \
    if (yydebug) {                                                             \
      YYFPRINTF(stderr, "%s ", Title);                                         \
      yy_symbol_print(stderr, Type, Value, parser);                            \
      YYFPRINTF(stderr, "\n");                                                 \
    }                                                                          \
  } while (YYID(0))

/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static void yy_symbol_value_print(FILE *yyoutput, int yytype,
                                  YYSTYPE const *const yyvaluep,
                                  vm_module_parser *parser)
#else
static void yy_symbol_value_print(yyoutput, yytype, yyvaluep,
                                  parser) FILE *yyoutput;
int yytype;
YYSTYPE const *const yyvaluep;
vm_module_parser *parser;
#endif
{
  FILE *yyo = yyoutput;
  YYUSE(yyo);
  if (!yyvaluep)
    return;
  YYUSE(parser);
#ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT(yyoutput, yytoknum[yytype], *yyvaluep);
#else
  YYUSE(yyoutput);
#endif
  switch (yytype) {
  default:
    break;
  }
}

/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static void yy_symbol_print(FILE *yyoutput, int yytype,
                            YYSTYPE const *const yyvaluep,
                            vm_module_parser *parser)
#else
static void yy_symbol_print(yyoutput, yytype, yyvaluep, parser) FILE *yyoutput;
int yytype;
YYSTYPE const *const yyvaluep;
vm_module_parser *parser;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF(yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF(yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print(yyoutput, yytype, yyvaluep, parser);
  YYFPRINTF(yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static void yy_stack_print(yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void yy_stack_print(yybottom, yytop) yytype_int16 *yybottom;
yytype_int16 *yytop;
#endif
{
  YYFPRINTF(stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++) {
    int yybot = *yybottom;
    YYFPRINTF(stderr, " %d", yybot);
  }
  YYFPRINTF(stderr, "\n");
}

#define YY_STACK_PRINT(Bottom, Top)                                            \
  do {                                                                         \
    if (yydebug)                                                               \
      yy_stack_print((Bottom), (Top));                                         \
  } while (YYID(0))

/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static void yy_reduce_print(YYSTYPE *yyvsp, int yyrule,
                            vm_module_parser *parser)
#else
static void yy_reduce_print(yyvsp, yyrule, parser) YYSTYPE *yyvsp;
int yyrule;
vm_module_parser *parser;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF(stderr, "Reducing stack by rule %d (line %lu):\n", yyrule - 1,
            yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++) {
    YYFPRINTF(stderr, "   $%d = ", yyi + 1);
    yy_symbol_print(stderr, yyrhs[yyprhs[yyrule] + yyi],
                    &(yyvsp[(yyi + 1) - (yynrhs)]), parser);
    YYFPRINTF(stderr, "\n");
  }
}

#define YY_REDUCE_PRINT(Rule)                                                  \
  do {                                                                         \
    if (yydebug)                                                               \
      yy_reduce_print(yyvsp, Rule, parser);                                    \
  } while (YYID(0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !VM_MODULE_PARSER_DEBUG */
#define YYDPRINTF(Args)
#define YY_SYMBOL_PRINT(Title, Type, Value, Location)
#define YY_STACK_PRINT(Bottom, Top)
#define YY_REDUCE_PRINT(Rule)
#endif /* !VM_MODULE_PARSER_DEBUG */

/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

#if YYERROR_VERBOSE

#ifndef yystrlen
#if defined __GLIBC__ && defined _STRING_H
#define yystrlen strlen
#else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static YYSIZE_T yystrlen(const char *yystr)
#else
static YYSIZE_T yystrlen(yystr) const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#endif
#endif

#ifndef yystpcpy
#if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#define yystpcpy stpcpy
#else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static char *yystpcpy(char *yydest, const char *yysrc)
#else
static char *yystpcpy(yydest, yysrc)
char *yydest;
const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T yytnamerr(char *yyres, const char *yystr) {
  if (*yystr == '"') {
    YYSIZE_T yyn = 0;
    char const *yyp = yystr;

    for (;;)
      switch (*++yyp) {
      case '\'':
      case ',':
        goto do_not_strip_quotes;

      case '\\':
        if (*++yyp != '\\')
          goto do_not_strip_quotes;
        /* Fall through.  */
      default:
        if (yyres)
          yyres[yyn] = *yyp;
        yyn++;
        break;

      case '"':
        if (yyres)
          yyres[yyn] = '\0';
        return yyn;
      }
  do_not_strip_quotes:;
  }

  if (!yyres)
    return yystrlen(yystr);

  return yystpcpy(yyres, yystr) - yyres;
}
#endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int yysyntax_error(YYSIZE_T *yymsg_alloc, char **yymsg,
                          yytype_int16 *yyssp, int yytoken) {
  YYSIZE_T yysize0 = yytnamerr(YY_NULL, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULL;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY) {
    int yyn = yypact[*yyssp];
    yyarg[yycount++] = yytname[yytoken];
    if (!yypact_value_is_default(yyn)) {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR &&
            !yytable_value_is_error(yytable[yyx + yyn])) {
          if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM) {
            yycount = 1;
            yysize = yysize0;
            break;
          }
          yyarg[yycount++] = yytname[yyx];
          {
            YYSIZE_T yysize1 = yysize + yytnamerr(YY_NULL, yytname[yyx]);
            if (!(yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
              return 2;
            yysize = yysize1;
          }
        }
    }
  }

  switch (yycount) {
#define YYCASE_(N, S)                                                          \
  case N:                                                                      \
    yyformat = S;                                                              \
    break
    YYCASE_(0, YY_("syntax error"));
    YYCASE_(1, YY_("syntax error, unexpected %s"));
    YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
    YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
    YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
    YYCASE_(5,
            YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
  }

  {
    YYSIZE_T yysize1 = yysize + yystrlen(yyformat);
    if (!(yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize) {
    *yymsg_alloc = 2 * yysize;
    if (!(yysize <= *yymsg_alloc && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
      *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
    return 1;
  }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount) {
        yyp += yytnamerr(yyp, yyarg[yyi++]);
        yyformat += 2;
      } else {
        yyp++;
        yyformat++;
      }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
static void yydestruct(const char *yymsg, int yytype, YYSTYPE *yyvaluep,
                       vm_module_parser *parser)
#else
static void yydestruct(yymsg, yytype, yyvaluep, parser) const char *yymsg;
int yytype;
YYSTYPE *yyvaluep;
vm_module_parser *parser;
#endif
{
  YYUSE(yyvaluep);
  YYUSE(parser);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT(yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype) {

  default:
    break;
  }
}

/* The lookahead symbol.  */
int yychar;

#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
#define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval YY_INITIAL_VALUE(yyval_default);

/* Number of syntax errors so far.  */
int yynerrs;

/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
int yyparse(void *YYPARSE_PARAM)
#else
int yyparse(YYPARSE_PARAM) void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ || defined __cplusplus ||       \
     defined _MSC_VER)
int yyparse(vm_module_parser *parser)
#else
int yyparse(parser)
vm_module_parser *parser;
#endif
#endif
{
  int yystate;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;

  /* The stacks and their tools:
     `yyss': related to states.
     `yyvs': related to semantic values.

     Refer to the stacks through separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs;
  YYSTYPE *yyvsp;

  YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N) (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

  /*------------------------------------------------------------.
  | yynewstate -- Push a new state, which is found in yystate.  |
  `------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp) {
    /* Get the current used size of the three stacks, in elements.  */
    YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
    {
      /* Give user a chance to reallocate the stack.  Use copies of
         these so that the &'s don't force the real ones into
         memory.  */
      YYSTYPE *yyvs1 = yyvs;
      yytype_int16 *yyss1 = yyss;

      /* Each stack pointer address is followed by the size of the
         data in use in that stack, in bytes.  This used to be a
         conditional around just the two extra args, but that might
         be undefined if yyoverflow is a macro.  */
      yyoverflow(YY_("memory exhausted"), &yyss1, yysize * sizeof(*yyssp),
                 &yyvs1, yysize * sizeof(*yyvsp), &yystacksize);

      yyss = yyss1;
      yyvs = yyvs1;
    }
#else /* no yyoverflow */
#ifndef YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    /* Extend the stack our own way.  */
    if (YYMAXDEPTH <= yystacksize)
      goto yyexhaustedlab;
    yystacksize *= 2;
    if (YYMAXDEPTH < yystacksize)
      yystacksize = YYMAXDEPTH;

    {
      yytype_int16 *yyss1 = yyss;
      union yyalloc *yyptr =
          (union yyalloc *)YYSTACK_ALLOC(YYSTACK_BYTES(yystacksize));
      if (!yyptr)
        goto yyexhaustedlab;
      YYSTACK_RELOCATE(yyss_alloc, yyss);
      YYSTACK_RELOCATE(yyvs_alloc, yyvs);
#undef YYSTACK_RELOCATE
      if (yyss1 != yyssa)
        YYSTACK_FREE(yyss1);
    }
#endif
#endif /* no yyoverflow */

    yyssp = yyss + yysize - 1;
    yyvsp = yyvs + yysize - 1;

    YYDPRINTF((stderr, "Stack size increased to %lu\n",
               (unsigned long int)yystacksize));

    if (yyss + yystacksize - 1 <= yyssp)
      YYABORT;
  }

  YYDPRINTF((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default(yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY) {
    YYDPRINTF((stderr, "Reading a token: "));
    yychar = YYLEX;
  }

  if (yychar <= YYEOF) {
    yychar = yytoken = YYEOF;
    YYDPRINTF((stderr, "Now at end of input.\n"));
  } else {
    yytoken = YYTRANSLATE(yychar);
    YY_SYMBOL_PRINT("Next token is", yytoken, &yylval, &yylloc);
  }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0) {
    if (yytable_value_is_error(yyn))
      goto yyerrlab;
    yyn = -yyn;
    goto yyreduce;
  }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;

/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;

/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1 - yylen];

  YY_REDUCE_PRINT(yyn);
  switch (yyn) {
  case 2:
/* Line 1792 of yacc.c  */
#line 104 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 3:
/* Line 1792 of yacc.c  */
#line 108 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 4:
/* Line 1792 of yacc.c  */
#line 109 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 5:
/* Line 1792 of yacc.c  */
#line 113 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment((yyvsp[(1) - (1)].string));
  } break;

  case 6:
/* Line 1792 of yacc.c  */
#line 119 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 7:
/* Line 1792 of yacc.c  */
#line 121 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 8:
/* Line 1792 of yacc.c  */
#line 127 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_end();
  } break;

  case 9:
/* Line 1792 of yacc.c  */
#line 128 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opCALL_EXT_METHOD, 0, 0, 0);
    function_end();
  } break;

  case 10:
/* Line 1792 of yacc.c  */
#line 129 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 11:
/* Line 1792 of yacc.c  */
#line 130 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 12:
/* Line 1792 of yacc.c  */
#line 134 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_return_type_set((yyvsp[(1) - (6)].string),
                             (yyvsp[(6) - (6)].string));
    trait_type_set((yyvsp[(1) - (6)].string), (yyvsp[(6) - (6)].string), false);
    track((yyvsp[(6) - (6)].string));
    (yyval.string) = (yyvsp[(1) - (6)].string);
  } break;

  case 13:
/* Line 1792 of yacc.c  */
#line 135 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_type = string_clone("void");
    function_return_type_set((yyvsp[(1) - (4)].string), vm_tmp_type);
    trait_type_set((yyvsp[(1) - (4)].string), vm_tmp_type, false);
    track(vm_tmp_type);
    (yyval.string) = (yyvsp[(1) - (4)].string);
  } break;

  case 14:
/* Line 1792 of yacc.c  */
#line 139 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_return_type_set((yyvsp[(2) - (7)].string),
                             (yyvsp[(7) - (7)].string));
    trait_type_set((yyvsp[(2) - (7)].string), (yyvsp[(7) - (7)].string), false);
    track((yyvsp[(7) - (7)].string));
    (yyval.string) = (yyvsp[(2) - (7)].string);
  } break;

  case 15:
/* Line 1792 of yacc.c  */
#line 140 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_return_type_set((yyvsp[(2) - (5)].string),
                             track(string_clone("void")));
    trait_type_set((yyvsp[(2) - (5)].string), string_clone("void"), false);
    (yyval.string) = (yyvsp[(2) - (5)].string);
  } break;

  case 16:
/* Line 1792 of yacc.c  */
#line 146 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    reset_registers();
    function_begin((yyvsp[(2) - (2)].string));
    (yyval.string) = (yyvsp[(2) - (2)].string);
  } break;

  case 17:
/* Line 1792 of yacc.c  */
#line 153 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 18:
/* Line 1792 of yacc.c  */
#line 154 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 19:
/* Line 1792 of yacc.c  */
#line 155 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 20:
/* Line 1792 of yacc.c  */
#line 159 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_parameter((yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string));
    emit(opSTACK_POP, 0, 0, (yyvsp[(1) - (3)].string));
    trait_type_set((yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string), false);
    track((yyvsp[(1) - (3)].string));
    track((yyvsp[(3) - (3)].string));
    (yyval.string) = (yyvsp[(1) - (3)].string);
  } break;

  case 21:
/* Line 1792 of yacc.c  */
#line 163 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 22:
/* Line 1792 of yacc.c  */
#line 164 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 23:
/* Line 1792 of yacc.c  */
#line 165 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 24:
/* Line 1792 of yacc.c  */
#line 169 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_parameter((yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string));
    trait_type_set((yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string), false);
    track((yyvsp[(1) - (3)].string));
    track((yyvsp[(3) - (3)].string));
    (yyval.string) = (yyvsp[(1) - (3)].string);
  } break;

  case 25:
/* Line 1792 of yacc.c  */
#line 176 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 26:
/* Line 1792 of yacc.c  */
#line 177 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 27:
/* Line 1792 of yacc.c  */
#line 186 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 28:
/* Line 1792 of yacc.c  */
#line 187 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSET_VAR, (yyvsp[(3) - (4)].string), 0, (yyvsp[(1) - (4)].string));
    trait_type_link((yyvsp[(3) - (4)].string), (yyvsp[(1) - (4)].string));
    track((yyvsp[(1) - (4)].string));
  } break;

  case 29:
/* Line 1792 of yacc.c  */
#line 188 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSET_VAR, (yyvsp[(3) - (4)].string), 0, (yyvsp[(1) - (4)].string));
    trait_type_link((yyvsp[(3) - (4)].string), (yyvsp[(1) - (4)].string));
    track((yyvsp[(1) - (4)].string));
  } break;

  case 30:
/* Line 1792 of yacc.c  */
#line 189 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSET_OBJ_VAR, (yyvsp[(1) - (6)].string), (yyvsp[(3) - (6)].string),
         (yyvsp[(5) - (6)].string));
    trait_type_link((yyvsp[(5) - (6)].string),
                    (yyvsp[(3) - (6)].string)); /* TODO */
  } break;

  case 31:
/* Line 1792 of yacc.c  */
#line 190 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSET_OBJ_VAR, (yyvsp[(1) - (6)].string), (yyvsp[(3) - (6)].string),
         (yyvsp[(5) - (6)].string));
    trait_type_link((yyvsp[(5) - (6)].string),
                    (yyvsp[(3) - (6)].string)); /* TODO */
  } break;

  case 32:
/* Line 1792 of yacc.c  */
#line 191 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 33:
/* Line 1792 of yacc.c  */
#line 192 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 34:
/* Line 1792 of yacc.c  */
#line 193 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSTACK_PUSH, (yyvsp[(2) - (3)].string), 0, 0);
    emit(opRETURN, 0, 0, 0);
  } break;

  case 35:
/* Line 1792 of yacc.c  */
#line 194 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSTACK_PUSH, (yyvsp[(2) - (3)].string), 0, 0);
    emit(opRETURN, 0, 0, 0);
  } break;

  case 36:
/* Line 1792 of yacc.c  */
#line 195 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opRETURN, 0, 0, 0);
  } break;

  case 37:
/* Line 1792 of yacc.c  */
#line 196 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opYIELD, (yyvsp[(3) - (5)].string), 0, 0);
    track((yyvsp[(3) - (5)].string));
  } break;

  case 38:
/* Line 1792 of yacc.c  */
#line 197 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    /* loop */
    comment("while_loop_end");
    vm_tmp_label = labels_pop();
    emit(opJUMP, 0, 0, labels_pop());
    emit(opLABEL, vm_tmp_label, 0, 0);
  } break;

  case 39:
/* Line 1792 of yacc.c  */
#line 201 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    /* for_increment_expression */
    emit_defer_pop();
    /* loop */
    comment("for_loop_end");
    vm_tmp_label = labels_pop();
    emit(opJUMP, 0, 0, labels_pop());
    emit(opLABEL, vm_tmp_label, 0, 0);
  } break;

  case 40:
/* Line 1792 of yacc.c  */
#line 208 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opLABEL, labels_pop(), 0, 0);
  } break;

  case 41:
/* Line 1792 of yacc.c  */
#line 209 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opLABEL, labels_pop(), 0, 0);
  } break;

  case 42:
/* Line 1792 of yacc.c  */
#line 210 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opLABEL, labels_pop(), 0, 0);
  } break;

  case 43:
/* Line 1792 of yacc.c  */
#line 211 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opLABEL, labels_pop(), 0, 0);
  } break;

  case 44:
/* Line 1792 of yacc.c  */
#line 212 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment((yyvsp[(1) - (1)].string));
  } break;

  case 45:
/* Line 1792 of yacc.c  */
#line 216 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opINC, (yyvsp[(1) - (2)].string), 0, 0);
    track((yyvsp[(1) - (2)].string));
  } break;

  case 46:
/* Line 1792 of yacc.c  */
#line 217 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opDEC, (yyvsp[(1) - (2)].string), 0, 0);
    track((yyvsp[(1) - (2)].string));
  } break;

  case 47:
/* Line 1792 of yacc.c  */
#line 218 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opADD, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         (yyvsp[(1) - (3)].string));
    trait_type_link((yyvsp[(3) - (3)].string), (yyvsp[(1) - (3)].string));
    track((yyvsp[(1) - (3)].string));
  } break;

  case 48:
/* Line 1792 of yacc.c  */
#line 219 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSUB, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         (yyvsp[(1) - (3)].string));
    trait_type_link((yyvsp[(3) - (3)].string), (yyvsp[(1) - (3)].string));
    track((yyvsp[(1) - (3)].string));
  } break;

  case 49:
/* Line 1792 of yacc.c  */
#line 223 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opCREATE_SET_VAR, (yyvsp[(4) - (4)].string), 0,
         (yyvsp[(2) - (4)].string));
    trait_type_link((yyvsp[(4) - (4)].string), (yyvsp[(2) - (4)].string));
    track((yyvsp[(2) - (4)].string));
  } break;

  case 50:
/* Line 1792 of yacc.c  */
#line 224 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opCREATE_SET_VAR, (yyvsp[(4) - (4)].string), 0,
         (yyvsp[(2) - (4)].string));
    trait_type_link((yyvsp[(4) - (4)].string), (yyvsp[(2) - (4)].string));
    track((yyvsp[(2) - (4)].string));
  } break;

  case 51:
/* Line 1792 of yacc.c  */
#line 228 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_label = create_label();
    emit(opJUMP_IF_FALSE, (yyvsp[(3) - (4)].string), 0, vm_tmp_label);
    labels_push(vm_tmp_label);
  } break;

  case 52:
/* Line 1792 of yacc.c  */
#line 232 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_label = create_label();
    emit(opJUMP, 0, 0, vm_tmp_label);
    emit(opLABEL, labels_pop(), 0, 0);
    labels_push(vm_tmp_label);
  } break;

  case 53:
/* Line 1792 of yacc.c  */
#line 236 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_label = create_label();
    emit(opJUMP_IF_FALSE, (yyvsp[(3) - (4)].string), 0, vm_tmp_label);
    labels_push(vm_tmp_label);
  } break;

  case 54:
/* Line 1792 of yacc.c  */
#line 240 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment("while_loop_begin");
    vm_tmp_label = create_label();
    emit(opLABEL, vm_tmp_label, 0, 0);
    labels_push(vm_tmp_label);
  } break;

  case 55:
/* Line 1792 of yacc.c  */
#line 246 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 56:
/* Line 1792 of yacc.c  */
#line 250 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment("for_loop_begin");
    vm_tmp_label = create_label();
    emit(opLABEL, vm_tmp_label, 0, 0);
    labels_push(vm_tmp_label);
  } break;

  case 57:
/* Line 1792 of yacc.c  */
#line 256 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_label = create_label();
    emit(opJUMP_IF_FALSE, (yyvsp[(1) - (1)].string), 0, vm_tmp_label);
    labels_push(vm_tmp_label);
    emit_defer_push_begin();
    comment("for_increment_expression");
  } break;

  case 58:
/* Line 1792 of yacc.c  */
#line 263 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit_defer_push_end();
  } break;

  case 59:
/* Line 1792 of yacc.c  */
#line 267 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(2) - (3)].string);
  } break;

  case 60:
/* Line 1792 of yacc.c  */
#line 268 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITIONAL_AND, (yyvsp[(1) - (3)].string),
         (yyvsp[(3) - (3)].string), vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 61:
/* Line 1792 of yacc.c  */
#line 269 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITIONAL_OR, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 62:
/* Line 1792 of yacc.c  */
#line 270 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 63:
/* Line 1792 of yacc.c  */
#line 274 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_GTE, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 64:
/* Line 1792 of yacc.c  */
#line 275 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_LTE, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 65:
/* Line 1792 of yacc.c  */
#line 276 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_GT, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 66:
/* Line 1792 of yacc.c  */
#line 277 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_LT, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 67:
/* Line 1792 of yacc.c  */
#line 278 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_NE, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 68:
/* Line 1792 of yacc.c  */
#line 279 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_EQ, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 69:
/* Line 1792 of yacc.c  */
#line 280 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opCONDITION_TRUE, (yyvsp[(1) - (1)].string), 0, vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 70:
/* Line 1792 of yacc.c  */
#line 281 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 71:
/* Line 1792 of yacc.c  */
#line 288 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opNOT, (yyvsp[(2) - (2)].string), 0, vm_tmp_var);
    trait_type_set(vm_tmp_var, "boolean", false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 72:
/* Line 1792 of yacc.c  */
#line 289 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    track((yyvsp[(1) - (1)].string));
    trait_type_set((yyvsp[(1) - (1)].string), "boolean", true);
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 73:
/* Line 1792 of yacc.c  */
#line 290 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    track((yyvsp[(1) - (1)].string));
    trait_type_set((yyvsp[(1) - (1)].string), "string", true);
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 74:
/* Line 1792 of yacc.c  */
#line 291 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opGET_OBJ_VAR, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    track((yyvsp[(1) - (3)].string));
    trait_type_link((yyvsp[(3) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 75:
/* Line 1792 of yacc.c  */
#line 292 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 76:
/* Line 1792 of yacc.c  */
#line 293 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opSTACK_POP, 0, 0, vm_tmp_var);
    trait_type_set(vm_tmp_var,
                   function_return_type_get((yyvsp[(1) - (1)].string)), false);
    (yyval.string) = vm_tmp_var;
  } break;

  case 77:
/* Line 1792 of yacc.c  */
#line 294 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    (yyval.string) = vm_tmp_var;
  } break;

  case 78:
/* Line 1792 of yacc.c  */
#line 295 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    (yyval.string) = vm_tmp_var;
  } break;

  case 79:
/* Line 1792 of yacc.c  */
#line 299 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    track((yyvsp[(1) - (1)].string));
    trait_type_set((yyvsp[(1) - (1)].string), "number", true);
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 80:
/* Line 1792 of yacc.c  */
#line 300 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    track((yyvsp[(1) - (1)].string));
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 81:
/* Line 1792 of yacc.c  */
#line 301 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(2) - (3)].string);
  } break;

  case 82:
/* Line 1792 of yacc.c  */
#line 302 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opADD, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_link((yyvsp[(1) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 83:
/* Line 1792 of yacc.c  */
#line 303 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opSUB, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_link((yyvsp[(1) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 84:
/* Line 1792 of yacc.c  */
#line 304 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opMUL, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_link((yyvsp[(1) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 85:
/* Line 1792 of yacc.c  */
#line 305 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opDIV, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_link((yyvsp[(1) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 86:
/* Line 1792 of yacc.c  */
#line 306 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opPOW, (yyvsp[(1) - (3)].string), (yyvsp[(3) - (3)].string),
         vm_tmp_var);
    trait_type_link((yyvsp[(1) - (3)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 87:
/* Line 1792 of yacc.c  */
#line 307 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    vm_tmp_var = create_register();
    emit(opNEG, (yyvsp[(2) - (2)].string), 0, vm_tmp_var);
    trait_type_link((yyvsp[(2) - (2)].string), vm_tmp_var);
    (yyval.string) = vm_tmp_var;
  } break;

  case 88:
/* Line 1792 of yacc.c  */
#line 318 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opCALL_METHOD, (yyvsp[(1) - (4)].string), 0, 0);
    trait_type_link((yyvsp[(1) - (4)].string),
                    function_return_type_get((yyvsp[(1) - (4)].string)));
    track((yyvsp[(1) - (4)].string));
    (yyval.string) = (yyvsp[(1) - (4)].string);
  } break;

  case 89:
/* Line 1792 of yacc.c  */
#line 319 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opCALL_OBJ_METHOD, (yyvsp[(1) - (6)].string),
         (yyvsp[(3) - (6)].string), 0);
    trait_type_link((yyvsp[(3) - (6)].string),
                    function_return_type_get((yyvsp[(3) - (6)].string)));
    track((yyvsp[(1) - (6)].string));
    track((yyvsp[(3) - (6)].string));
    (yyval.string) = (yyvsp[(3) - (6)].string); /* Needs fix for $1.$3*/
  } break;

  case 90:
/* Line 1792 of yacc.c  */
#line 326 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSTACK_PUSH, (yyvsp[(1) - (3)].string), 0, 0);
  } break;

  case 91:
/* Line 1792 of yacc.c  */
#line 327 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSTACK_PUSH, (yyvsp[(1) - (1)].string), 0, 0);
  } break;

  case 92:
/* Line 1792 of yacc.c  */
#line 328 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
  } break;

  case 93:
/* Line 1792 of yacc.c  */
#line 336 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    emit(opSTACK_PUSH, (yyvsp[(4) - (4)].string), 0, 0);
    emit(opRETURN, 0, 0, 0);
    function_end();
  } break;

  case 94:
/* Line 1792 of yacc.c  */
#line 342 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_return_type_set((yyvsp[(1) - (4)].string),
                             track(string_clone("number")));
    (yyval.string) = (yyvsp[(1) - (4)].string);
  } break;

  case 98:
/* Line 1792 of yacc.c  */
#line 352 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    function_parameter((yyvsp[(1) - (1)].string), "number");
    emit(opSTACK_POP, 0, 0, (yyvsp[(1) - (1)].string));
    trait_type_set((yyvsp[(1) - (1)].string), "number", false);
    track((yyvsp[(1) - (1)].string));
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 99:
/* Line 1792 of yacc.c  */
#line 356 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    reset_registers();
    function_begin((yyvsp[(1) - (1)].string));
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 100:
/* Line 1792 of yacc.c  */
#line 360 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 101:
/* Line 1792 of yacc.c  */
#line 361 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(3) - (7)].string);
  } break;

  case 102:
/* Line 1792 of yacc.c  */
#line 363 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    (yyval.string) = (yyvsp[(1) - (1)].string);
  } break;

  case 103:
/* Line 1792 of yacc.c  */
#line 367 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment("sum-of: end");
    comment("sum-of: summation");
    emit(opADD, (yyvsp[(1) - (2)].string), (yyvsp[(2) - (2)].string),
         (yyvsp[(1) - (2)].string));
    trait_type_link((yyvsp[(1) - (2)].string), (yyvsp[(2) - (2)].string));

    /* We should have our summation iterator stashed away */
    emit_defer_pop();

    /* Perform the summation loop */
    vm_tmp_label = labels_pop();
    emit(opJUMP, 0, 0, labels_pop());
    emit(opLABEL, vm_tmp_label, 0, 0);

    /* Return the variable with the summation */
    (yyval.string) = (yyvsp[(1) - (2)].string);
  } break;

  case 104:
/* Line 1792 of yacc.c  */
#line 385 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
  {
    comment("sum-of: begin");

    /* Create the initial start of the summation loop */
    emit(opCREATE_SET_VAR, (yyvsp[(5) - (8)].string), 0,
         (yyvsp[(3) - (8)].string));
    trait_type_link((yyvsp[(5) - (8)].string), (yyvsp[(3) - (8)].string));
    track((yyvsp[(3) - (8)].string));

    /* When we execute the expression we want to sum it */
    vm_tmp_var = create_register();
    trait_type_link((yyvsp[(3) - (8)].string), vm_tmp_var);

    /* Return the variable that has summation value */
    (yyval.string) = vm_tmp_var;

    /* The label where we will loop back to */
    vm_tmp_label = create_label();
    emit(opLABEL, vm_tmp_label, 0, 0);
    labels_push(vm_tmp_label);

    /* Has the summation limit been reached? */
    vm_tmp_var = create_register();
    emit(opCONDITION_LT, (yyvsp[(3) - (8)].string), (yyvsp[(8) - (8)].string),
         vm_tmp_var);
    trait_type_set((yyvsp[(8) - (8)].string), "number", false);
    trait_type_set(vm_tmp_var, "boolean", false);

    /* If yes, the exit the loop */
    vm_tmp_label = create_label();
    emit(opJUMP_IF_FALSE, vm_tmp_var, 0, vm_tmp_label);
    labels_push(vm_tmp_label);
    comment("apply-to: begin");

    /* We want to perform increment after the summation expression is completed,
     * so defer these instructions */
    emit_defer_push_begin();
    comment("apply-to: end");
    comment("sum-of: increment");
    emit(opINC, (yyvsp[(3) - (8)].string), 0, 0);
    emit_defer_push_end();
  } break;

/* Line 1792 of yacc.c  */
#line 2366 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.tab.cpp"
  default:
    break;
  }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;

/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE(yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus) {
    ++yynerrs;
#if !YYERROR_VERBOSE
    yyerror(parser, YY_("syntax error"));
#else
#define YYSYNTAX_ERROR yysyntax_error(&yymsg_alloc, &yymsg, yyssp, yytoken)
    {
      char const *yymsgp = YY_("syntax error");
      int yysyntax_error_status;
      yysyntax_error_status = YYSYNTAX_ERROR;
      if (yysyntax_error_status == 0)
        yymsgp = yymsg;
      else if (yysyntax_error_status == 1) {
        if (yymsg != yymsgbuf)
          YYSTACK_FREE(yymsg);
        yymsg = (char *)YYSTACK_ALLOC(yymsg_alloc);
        if (!yymsg) {
          yymsg = yymsgbuf;
          yymsg_alloc = sizeof yymsgbuf;
          yysyntax_error_status = 2;
        } else {
          yysyntax_error_status = YYSYNTAX_ERROR;
          yymsgp = yymsg;
        }
      }
      yyerror(parser, yymsgp);
      if (yysyntax_error_status == 2)
        goto yyexhaustedlab;
    }
#undef YYSYNTAX_ERROR
#endif
  }

  if (yyerrstatus == 3) {
    /* If just tried and failed to reuse lookahead token after an
       error, discard it.  */

    if (yychar <= YYEOF) {
      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        YYABORT;
    } else {
      yydestruct("Error: discarding", yytoken, &yylval, parser);
      yychar = YYEMPTY;
    }
  }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;

/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
    goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;

/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3; /* Each real token shifted decrements this.  */

  for (;;) {
    yyn = yypact[yystate];
    if (!yypact_value_is_default(yyn)) {
      yyn += YYTERROR;
      if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR) {
        yyn = yytable[yyn];
        if (0 < yyn)
          break;
      }
    }

    /* Pop the current state because it cannot handle the error token.  */
    if (yyssp == yyss)
      YYABORT;

    yydestruct("Error: popping", yystos[yystate], yyvsp, parser);
    YYPOPSTACK(1);
    yystate = *yyssp;
    YY_STACK_PRINT(yyss, yyssp);
  }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Shift the error token.  */
  YY_SYMBOL_PRINT("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;

/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror(parser, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY) {
    /* Make sure we have latest lookahead translation.  See comments at
       user semantic actions for why this is necessary.  */
    yytoken = YYTRANSLATE(yychar);
    yydestruct("Cleanup: discarding lookahead", yytoken, &yylval, parser);
  }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK(yylen);
  YY_STACK_PRINT(yyss, yyssp);
  while (yyssp != yyss) {
    yydestruct("Cleanup: popping", yystos[*yyssp], yyvsp, parser);
    YYPOPSTACK(1);
  }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE(yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE(yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID(yyresult);
}

/* Line 2055 of yacc.c  */
#line 470 "D:/dev/projects/enjector/microgpt-vm/main/core/src/vm_module_parser.y"
