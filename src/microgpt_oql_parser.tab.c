/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         OQL_PARSER_STYPE
/* Substitute the variable and function names.  */
#define yyparse         oql_parser_parse
#define yylex           oql_parser_lex
#define yyerror         oql_parser_error
#define yydebug         oql_parser_debug
#define yynerrs         oql_parser_nerrs
#define yylval          oql_parser_lval
#define yychar          oql_parser_char

/* First part of user prologue.  */
#line 13 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"

    #include "microgpt_oql.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    extern int oql_parser_lex(oql_parser *parser);
    extern void oql_parser_error(oql_parser *parser, const char *msg);

    /* AST helpers — defined in microgpt_oql.c as inlines for the grammar. */
    OqlStmt *oql_y_train(char *name, OqlSource on, OqlKV *with);
    OqlStmt *oql_y_compose(char *name, OqlNameList *from, OqlKV *with);
    OqlStmt *oql_y_run(char *name, OqlKV *with);
    OqlStmt *oql_y_evaluate(char *name, OqlSource against, char *metric, char *report);
    OqlStmt *oql_y_verify(OqlVerifySubjectKind k, char *subject, OqlPredicate *where);
    OqlStmt *oql_y_audit(OqlSource a, OqlSource b, char *thr, char *report);
    OqlStmt *oql_y_create_behaviour(char *name, char *vm_body);
    OqlStmt *oql_y_create_organelle(char *name, char *ckpt, OqlKV *bindings);
    OqlStmt *oql_y_create_corpus(char *name, char *path);
    /* E12 — CREATE CORPUS <name> FROM LLM '<model>'[@'<endpoint>']
     *           PROMPT '<text>' [WITH (...)] [VERIFY_VIA pipeline_ir]
     *           [AUDIT_AGAINST <held_out_name>]; */
    OqlStmt *oql_y_create_corpus_llm(char *name, char *model_id,
                                     char *endpoint_url, char *prompt,
                                     OqlKV *with_kv, int verify_via_pipeline_ir,
                                     char *audit_held_out, OqlKV *audit_with);
    /* E15 — CREATE CORPUS <name> FROM ORACLE '<path>' [WITH (...)] [PROMPT '<text>']; */
    OqlStmt *oql_y_create_corpus_oracle(char *name, char *oracle_path,
                                        OqlKV *with_kv, char *prompt);
    OqlKV   *oql_y_kv(char *key, char *val);
    OqlKV   *oql_y_kv_concat(OqlKV *head, OqlKV *tail);
    OqlNameList *oql_y_name(char *n);
    OqlNameList *oql_y_name_concat(OqlNameList *head, OqlNameList *tail);
    OqlPredicate *oql_y_pred(char *lhs, OqlOp op, char *rhs);
    void oql_y_append(oql_parser *p, OqlStmt *s);

#line 116 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "microgpt_oql_parser.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_T_TRAIN = 3,                    /* T_TRAIN  */
  YYSYMBOL_T_COMPOSE = 4,                  /* T_COMPOSE  */
  YYSYMBOL_T_RUN = 5,                      /* T_RUN  */
  YYSYMBOL_T_EVALUATE = 6,                 /* T_EVALUATE  */
  YYSYMBOL_T_VERIFY = 7,                   /* T_VERIFY  */
  YYSYMBOL_T_AUDIT = 8,                    /* T_AUDIT  */
  YYSYMBOL_T_CREATE = 9,                   /* T_CREATE  */
  YYSYMBOL_T_BEHAVIOUR = 10,               /* T_BEHAVIOUR  */
  YYSYMBOL_T_ORGANELLE = 11,               /* T_ORGANELLE  */
  YYSYMBOL_T_CHECKPOINT = 12,              /* T_CHECKPOINT  */
  YYSYMBOL_T_VM = 13,                      /* T_VM  */
  YYSYMBOL_T_FILE = 14,                    /* T_FILE  */
  YYSYMBOL_T_ON = 15,                      /* T_ON  */
  YYSYMBOL_T_WITH = 16,                    /* T_WITH  */
  YYSYMBOL_T_AGAINST = 17,                 /* T_AGAINST  */
  YYSYMBOL_T_USING = 18,                   /* T_USING  */
  YYSYMBOL_T_WHERE = 19,                   /* T_WHERE  */
  YYSYMBOL_T_AS = 20,                      /* T_AS  */
  YYSYMBOL_T_FROM = 21,                    /* T_FROM  */
  YYSYMBOL_T_GRAPH = 22,                   /* T_GRAPH  */
  YYSYMBOL_T_CORPUS = 23,                  /* T_CORPUS  */
  YYSYMBOL_T_REPORT = 24,                  /* T_REPORT  */
  YYSYMBOL_T_METRIC = 25,                  /* T_METRIC  */
  YYSYMBOL_T_THRESHOLDS = 26,              /* T_THRESHOLDS  */
  YYSYMBOL_T_LT = 27,                      /* T_LT  */
  YYSYMBOL_T_LE = 28,                      /* T_LE  */
  YYSYMBOL_T_EQ = 29,                      /* T_EQ  */
  YYSYMBOL_T_NE = 30,                      /* T_NE  */
  YYSYMBOL_T_GE = 31,                      /* T_GE  */
  YYSYMBOL_T_GT = 32,                      /* T_GT  */
  YYSYMBOL_T_LLM = 33,                     /* T_LLM  */
  YYSYMBOL_T_PROMPT = 34,                  /* T_PROMPT  */
  YYSYMBOL_T_VERIFY_VIA = 35,              /* T_VERIFY_VIA  */
  YYSYMBOL_T_AUDIT_AGAINST = 36,           /* T_AUDIT_AGAINST  */
  YYSYMBOL_T_PIPELINE_IR = 37,             /* T_PIPELINE_IR  */
  YYSYMBOL_T_AT = 38,                      /* T_AT  */
  YYSYMBOL_T_ORACLE = 39,                  /* T_ORACLE  */
  YYSYMBOL_T_IDENT = 40,                   /* T_IDENT  */
  YYSYMBOL_T_STRING = 41,                  /* T_STRING  */
  YYSYMBOL_T_NUMBER = 42,                  /* T_NUMBER  */
  YYSYMBOL_T_GRAPH_BLOCK = 43,             /* T_GRAPH_BLOCK  */
  YYSYMBOL_T_VM_BODY = 44,                 /* T_VM_BODY  */
  YYSYMBOL_45_ = 45,                       /* ';'  */
  YYSYMBOL_46_ = 46,                       /* '('  */
  YYSYMBOL_47_ = 47,                       /* ')'  */
  YYSYMBOL_48_ = 48,                       /* ','  */
  YYSYMBOL_YYACCEPT = 49,                  /* $accept  */
  YYSYMBOL_script = 50,                    /* script  */
  YYSYMBOL_stmt = 51,                      /* stmt  */
  YYSYMBOL_train_stmt = 52,                /* train_stmt  */
  YYSYMBOL_opt_on = 53,                    /* opt_on  */
  YYSYMBOL_compose_stmt = 54,              /* compose_stmt  */
  YYSYMBOL_run_stmt = 55,                  /* run_stmt  */
  YYSYMBOL_evaluate_stmt = 56,             /* evaluate_stmt  */
  YYSYMBOL_opt_metric = 57,                /* opt_metric  */
  YYSYMBOL_opt_report = 58,                /* opt_report  */
  YYSYMBOL_verify_stmt = 59,               /* verify_stmt  */
  YYSYMBOL_opt_where = 60,                 /* opt_where  */
  YYSYMBOL_predicate = 61,                 /* predicate  */
  YYSYMBOL_op = 62,                        /* op  */
  YYSYMBOL_audit_stmt = 63,                /* audit_stmt  */
  YYSYMBOL_opt_thresholds = 64,            /* opt_thresholds  */
  YYSYMBOL_create_stmt = 65,               /* create_stmt  */
  YYSYMBOL_create_behaviour_stmt = 66,     /* create_behaviour_stmt  */
  YYSYMBOL_create_organelle_stmt = 67,     /* create_organelle_stmt  */
  YYSYMBOL_create_corpus_stmt = 68,        /* create_corpus_stmt  */
  YYSYMBOL_create_corpus_llm_stmt = 69,    /* create_corpus_llm_stmt  */
  YYSYMBOL_opt_llm_endpoint = 70,          /* opt_llm_endpoint  */
  YYSYMBOL_opt_llm_audit_with = 71,        /* opt_llm_audit_with  */
  YYSYMBOL_create_corpus_oracle_stmt = 72, /* create_corpus_oracle_stmt  */
  YYSYMBOL_opt_oracle_prompt = 73,         /* opt_oracle_prompt  */
  YYSYMBOL_opt_with_bindings = 74,         /* opt_with_bindings  */
  YYSYMBOL_binding_list = 75,              /* binding_list  */
  YYSYMBOL_binding = 76,                   /* binding  */
  YYSYMBOL_opt_with = 77,                  /* opt_with  */
  YYSYMBOL_kv_list = 78,                   /* kv_list  */
  YYSYMBOL_kv = 79,                        /* kv  */
  YYSYMBOL_value = 80,                     /* value  */
  YYSYMBOL_source = 81,                    /* source  */
  YYSYMBOL_name_list = 82                  /* name_list  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined OQL_PARSER_STYPE_IS_TRIVIAL && OQL_PARSER_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   119

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  49
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  34
/* YYNRULES -- Number of rules.  */
#define YYNRULES  73
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  143

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   299


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      46,    47,     2,     2,    48,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    45,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44
};

#if OQL_PARSER_DEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    87,    87,    87,    90,    91,    92,    93,    94,    95,
      96,   101,   104,   105,   110,   115,   120,   124,   125,   128,
     129,   134,   135,   136,   139,   140,   143,   146,   146,   147,
     147,   148,   148,   153,   157,   158,   163,   164,   165,   166,
     167,   170,   174,   176,   183,   195,   198,   202,   206,   213,
     214,   217,   218,   228,   232,   233,   237,   238,   241,   242,
     245,   250,   251,   254,   255,   258,   261,   261,   261,   264,
     265,   266,   269,   270
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if OQL_PARSER_DEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "T_TRAIN", "T_COMPOSE",
  "T_RUN", "T_EVALUATE", "T_VERIFY", "T_AUDIT", "T_CREATE", "T_BEHAVIOUR",
  "T_ORGANELLE", "T_CHECKPOINT", "T_VM", "T_FILE", "T_ON", "T_WITH",
  "T_AGAINST", "T_USING", "T_WHERE", "T_AS", "T_FROM", "T_GRAPH",
  "T_CORPUS", "T_REPORT", "T_METRIC", "T_THRESHOLDS", "T_LT", "T_LE",
  "T_EQ", "T_NE", "T_GE", "T_GT", "T_LLM", "T_PROMPT", "T_VERIFY_VIA",
  "T_AUDIT_AGAINST", "T_PIPELINE_IR", "T_AT", "T_ORACLE", "T_IDENT",
  "T_STRING", "T_NUMBER", "T_GRAPH_BLOCK", "T_VM_BODY", "';'", "'('",
  "')'", "','", "$accept", "script", "stmt", "train_stmt", "opt_on",
  "compose_stmt", "run_stmt", "evaluate_stmt", "opt_metric", "opt_report",
  "verify_stmt", "opt_where", "predicate", "op", "audit_stmt",
  "opt_thresholds", "create_stmt", "create_behaviour_stmt",
  "create_organelle_stmt", "create_corpus_stmt", "create_corpus_llm_stmt",
  "opt_llm_endpoint", "opt_llm_audit_with", "create_corpus_oracle_stmt",
  "opt_oracle_prompt", "opt_with_bindings", "binding_list", "binding",
  "opt_with", "kv_list", "kv", "value", "source", "name_list", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-52)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int8 yypact[] =
{
     -52,    31,   -52,   -25,    -8,    10,    11,   -18,   -16,     3,
       8,   -52,   -52,   -52,   -52,   -52,   -52,   -52,   -52,   -52,
     -52,   -52,   -52,    -3,     9,    36,    37,    12,    38,    38,
      15,   -52,   -52,    41,    19,    20,    21,   -52,   -16,    36,
      22,    23,   -52,   -16,    38,    24,   -52,   -52,   -52,   -16,
      45,   -10,    46,   -52,   -52,   -52,   -15,    39,    18,   -52,
      51,   -52,    14,   -52,    52,    59,    27,    62,   -52,   -12,
      35,   -52,     7,    23,    53,    55,   -52,   -52,   -52,   -52,
     -52,   -52,     7,    50,    55,    33,    40,    42,    43,    44,
      47,   -52,   -52,   -52,   -52,   -52,   -52,    54,    61,   -52,
     -52,    49,   -52,   -52,    57,   -28,   -52,    66,   -52,    58,
      36,   -52,    56,   -52,    60,   -52,    40,   -52,    63,    64,
      65,   -52,   -52,   -52,   -52,    67,    68,   -52,    36,   -52,
     -27,    69,    70,    71,    75,    72,    73,   -52,    75,    23,
     -52,   -19,   -52
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       2,     0,     1,     0,     0,     0,     0,     0,     0,     0,
       0,     4,     5,     6,     7,     8,     9,    10,    36,    37,
      38,    39,    40,    12,     0,    61,     0,     0,    24,    24,
       0,    71,    70,     0,     0,     0,     0,     3,     0,    61,
       0,     0,    15,     0,    24,     0,    23,    22,    69,     0,
       0,    56,     0,    13,    11,    72,    61,     0,    62,    63,
      17,    21,     0,    25,    34,     0,     0,     0,    43,     0,
       0,    14,     0,     0,     0,    19,    27,    28,    29,    30,
      31,    32,     0,     0,    19,     0,     0,     0,     0,     0,
       0,    73,    68,    66,    67,    65,    64,     0,     0,    16,
      26,     0,    33,    41,     0,     0,    58,    56,    44,    49,
      61,    18,     0,    35,     0,    57,     0,    42,     0,     0,
      54,    20,    60,    59,    50,     0,     0,    53,    61,    55,
      45,     0,     0,    46,    51,     0,     0,    47,    51,     0,
      48,     0,    52
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -52,   -52,   -52,   -52,   -52,   -52,   -52,   -52,   -52,    17,
     -52,   -26,   -52,   -52,   -52,   -52,   -52,   -52,   -52,   -52,
     -52,   -52,   -51,   -52,   -52,   -14,   -52,   -24,   -39,   -44,
      29,    32,   -33,   -52
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     1,    10,    11,    39,    12,    13,    14,    75,    99,
      15,    46,    63,    82,    16,    84,    17,    18,    19,    20,
      21,   119,   137,    22,   127,    68,   105,   106,    42,    58,
      59,    95,    33,    56
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      54,    41,    88,    47,    27,    53,    66,    30,   131,   132,
      60,    67,    38,    34,    35,    23,    64,    71,    61,   115,
     116,    89,    28,    29,    31,    32,    36,    90,   142,    73,
      40,     2,    24,    70,     3,     4,     5,     6,     7,     8,
       9,    76,    77,    78,    79,    80,    81,    92,    93,    94,
      25,    26,    41,    37,    43,    44,    48,    45,    49,    50,
      51,    52,    55,    57,    62,    65,    73,    69,    72,    74,
      83,   120,    85,    86,    87,    91,   101,   103,    97,    98,
     104,   112,    66,   107,   108,   109,   114,   140,   110,   130,
     113,   136,   123,   117,   111,   141,   118,   121,   125,   126,
     122,   102,    96,     0,   124,     0,   133,   135,   128,   129,
     134,     0,   138,     0,   100,     0,     0,     0,     0,   139
};

static const yytype_int16 yycheck[] =
{
      39,    16,    14,    29,    22,    38,    16,    23,    35,    36,
      43,    21,    15,    10,    11,    40,    49,    56,    44,    47,
      48,    33,    40,    41,    40,    41,    23,    39,    47,    48,
      21,     0,    40,    48,     3,     4,     5,     6,     7,     8,
       9,    27,    28,    29,    30,    31,    32,    40,    41,    42,
      40,    40,    16,    45,    17,    43,    41,    19,    17,    40,
      40,    40,    40,    40,    40,    20,    48,    21,    29,    18,
      18,   110,    13,    46,    12,    40,    26,    44,    25,    24,
      40,    20,    16,    41,    41,    41,    29,   138,    41,   128,
      41,    16,   116,   107,    40,   139,    38,    41,    34,    34,
      40,    84,    73,    -1,    41,    -1,    37,    36,    41,    41,
      40,    -1,    40,    -1,    82,    -1,    -1,    -1,    -1,    46
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    50,     0,     3,     4,     5,     6,     7,     8,     9,
      51,    52,    54,    55,    56,    59,    63,    65,    66,    67,
      68,    69,    72,    40,    40,    40,    40,    22,    40,    41,
      23,    40,    41,    81,    10,    11,    23,    45,    15,    53,
      21,    16,    77,    17,    43,    19,    60,    60,    41,    17,
      40,    40,    40,    81,    77,    40,    82,    40,    78,    79,
      81,    60,    40,    61,    81,    20,    16,    21,    74,    21,
      48,    77,    29,    48,    18,    57,    27,    28,    29,    30,
      31,    32,    62,    18,    64,    13,    46,    12,    14,    33,
      39,    40,    40,    41,    42,    80,    79,    25,    24,    58,
      80,    26,    58,    44,    40,    75,    76,    41,    41,    41,
      41,    40,    20,    41,    29,    47,    48,    74,    38,    70,
      77,    41,    40,    76,    41,    34,    34,    73,    41,    41,
      77,    35,    36,    37,    40,    36,    16,    71,    40,    46,
      71,    78,    47
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    49,    50,    50,    51,    51,    51,    51,    51,    51,
      51,    52,    53,    53,    54,    55,    56,    57,    57,    58,
      58,    59,    59,    59,    60,    60,    61,    62,    62,    62,
      62,    62,    62,    63,    64,    64,    65,    65,    65,    65,
      65,    66,    67,    67,    68,    69,    69,    69,    69,    70,
      70,    71,    71,    72,    73,    73,    74,    74,    75,    75,
      76,    77,    77,    78,    78,    79,    80,    80,    80,    81,
      81,    81,    82,    82
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     3,     1,     1,     1,     1,     1,     1,
       1,     4,     0,     2,     5,     3,     6,     0,     3,     0,
       3,     4,     3,     3,     0,     2,     3,     1,     1,     1,
       1,     1,     1,     6,     0,     3,     1,     1,     1,     1,
       1,     6,     7,     4,     6,    10,    12,    13,    15,     0,
       2,     0,     4,     8,     0,     2,     0,     4,     1,     3,
       3,     0,     2,     1,     3,     3,     1,     1,     1,     2,
       1,     1,     1,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = OQL_PARSER_EMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == OQL_PARSER_EMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (parser, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use OQL_PARSER_error or OQL_PARSER_UNDEF. */
#define YYERRCODE OQL_PARSER_UNDEF


/* Enable debugging if requested.  */
#if OQL_PARSER_DEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, parser); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, oql_parser *parser)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (parser);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, oql_parser *parser)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep, parser);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule, oql_parser *parser)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)], parser);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, parser); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !OQL_PARSER_DEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !OQL_PARSER_DEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, oql_parser *parser)
{
  YY_USE (yyvaluep);
  YY_USE (parser);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (oql_parser *parser)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = OQL_PARSER_EMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


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
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == OQL_PARSER_EMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (parser);
    }

  if (yychar <= OQL_PARSER_EOF)
    {
      yychar = OQL_PARSER_EOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == OQL_PARSER_error)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = OQL_PARSER_UNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = OQL_PARSER_EMPTY;
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
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 3: /* script: script stmt ';'  */
#line 87 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                       { oql_y_append(parser, (yyvsp[-1].stmt)); }
#line 1282 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 4: /* stmt: train_stmt  */
#line 90 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1288 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 5: /* stmt: compose_stmt  */
#line 91 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1294 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 6: /* stmt: run_stmt  */
#line 92 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1300 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 7: /* stmt: evaluate_stmt  */
#line 93 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1306 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 8: /* stmt: verify_stmt  */
#line 94 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1312 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 9: /* stmt: audit_stmt  */
#line 95 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1318 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 10: /* stmt: create_stmt  */
#line 96 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1324 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 11: /* train_stmt: T_TRAIN T_IDENT opt_on opt_with  */
#line 101 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_train((yyvsp[-2].str), (yyvsp[-1].src), (yyvsp[0].kv)); }
#line 1330 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 12: /* opt_on: %empty  */
#line 104 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { OqlSource z = {0, NULL}; (yyval.src) = z; }
#line 1336 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 13: /* opt_on: T_ON source  */
#line 105 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.src) = (yyvsp[0].src); }
#line 1342 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 14: /* compose_stmt: T_COMPOSE T_IDENT T_FROM name_list opt_with  */
#line 110 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_compose((yyvsp[-3].str), (yyvsp[-1].names), (yyvsp[0].kv)); }
#line 1348 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 15: /* run_stmt: T_RUN T_IDENT opt_with  */
#line 115 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_run((yyvsp[-1].str), (yyvsp[0].kv)); }
#line 1354 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 16: /* evaluate_stmt: T_EVALUATE T_IDENT T_AGAINST source opt_metric opt_report  */
#line 121 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_evaluate((yyvsp[-4].str), (yyvsp[-2].src), (yyvsp[-1].str), (yyvsp[0].str)); }
#line 1360 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 17: /* opt_metric: %empty  */
#line 124 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = NULL; }
#line 1366 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 18: /* opt_metric: T_USING T_METRIC T_IDENT  */
#line 125 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = (yyvsp[0].str); }
#line 1372 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 19: /* opt_report: %empty  */
#line 128 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = NULL; }
#line 1378 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 20: /* opt_report: T_REPORT T_AS T_STRING  */
#line 129 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = (yyvsp[0].str); }
#line 1384 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 21: /* verify_stmt: T_VERIFY T_GRAPH T_GRAPH_BLOCK opt_where  */
#line 134 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_verify(OQL_VS_GRAPH, (yyvsp[-1].str), (yyvsp[0].pred)); }
#line 1390 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 22: /* verify_stmt: T_VERIFY T_STRING opt_where  */
#line 135 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_verify(OQL_VS_PATH,  (yyvsp[-1].str), (yyvsp[0].pred)); }
#line 1396 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 23: /* verify_stmt: T_VERIFY T_IDENT opt_where  */
#line 136 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.stmt) = oql_y_verify(OQL_VS_NAME,  (yyvsp[-1].str), (yyvsp[0].pred)); }
#line 1402 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 24: /* opt_where: %empty  */
#line 139 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.pred) = NULL; }
#line 1408 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 25: /* opt_where: T_WHERE predicate  */
#line 140 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.pred) = (yyvsp[0].pred); }
#line 1414 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 26: /* predicate: T_IDENT op value  */
#line 143 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.pred) = oql_y_pred((yyvsp[-2].str), (OqlOp)(yyvsp[-1].op), (yyvsp[0].str)); }
#line 1420 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 27: /* op: T_LT  */
#line 146 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
           { (yyval.op) = OQL_OP_LT; }
#line 1426 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 28: /* op: T_LE  */
#line 146 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                      { (yyval.op) = OQL_OP_LE; }
#line 1432 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 29: /* op: T_EQ  */
#line 147 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
           { (yyval.op) = OQL_OP_EQ; }
#line 1438 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 30: /* op: T_NE  */
#line 147 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                      { (yyval.op) = OQL_OP_NE; }
#line 1444 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 31: /* op: T_GE  */
#line 148 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
           { (yyval.op) = OQL_OP_GE; }
#line 1450 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 32: /* op: T_GT  */
#line 148 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                      { (yyval.op) = OQL_OP_GT; }
#line 1456 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 33: /* audit_stmt: T_AUDIT source T_AGAINST source opt_thresholds opt_report  */
#line 154 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_audit((yyvsp[-4].src), (yyvsp[-2].src), (yyvsp[-1].str), (yyvsp[0].str)); }
#line 1462 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 34: /* opt_thresholds: %empty  */
#line 157 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = NULL; }
#line 1468 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 35: /* opt_thresholds: T_USING T_THRESHOLDS T_STRING  */
#line 158 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.str) = (yyvsp[0].str); }
#line 1474 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 36: /* create_stmt: create_behaviour_stmt  */
#line 163 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1480 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 37: /* create_stmt: create_organelle_stmt  */
#line 164 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1486 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 38: /* create_stmt: create_corpus_stmt  */
#line 165 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1492 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 39: /* create_stmt: create_corpus_llm_stmt  */
#line 166 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1498 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 40: /* create_stmt: create_corpus_oracle_stmt  */
#line 167 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.stmt) = (yyvsp[0].stmt); }
#line 1504 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 41: /* create_behaviour_stmt: T_CREATE T_BEHAVIOUR T_IDENT T_AS T_VM T_VM_BODY  */
#line 171 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_behaviour((yyvsp[-3].str), (yyvsp[0].str)); }
#line 1510 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 42: /* create_organelle_stmt: T_CREATE T_ORGANELLE T_IDENT T_FROM T_CHECKPOINT T_STRING opt_with_bindings  */
#line 175 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_organelle((yyvsp[-4].str), (yyvsp[-1].str), (yyvsp[0].kv)); }
#line 1516 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 43: /* create_organelle_stmt: T_CREATE T_ORGANELLE T_IDENT opt_with_bindings  */
#line 177 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_organelle((yyvsp[-1].str), NULL, (yyvsp[0].kv)); }
#line 1522 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 44: /* create_corpus_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_FILE T_STRING  */
#line 184 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus((yyvsp[-3].str), (yyvsp[0].str)); }
#line 1528 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 45: /* create_corpus_llm_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_LLM T_STRING opt_llm_endpoint T_PROMPT T_STRING opt_with  */
#line 197 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus_llm((yyvsp[-7].str), (yyvsp[-4].str), (yyvsp[-3].str), (yyvsp[-1].str), (yyvsp[0].kv), 0, NULL, NULL); }
#line 1534 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 46: /* create_corpus_llm_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_LLM T_STRING opt_llm_endpoint T_PROMPT T_STRING opt_with T_VERIFY_VIA T_PIPELINE_IR  */
#line 201 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus_llm((yyvsp[-9].str), (yyvsp[-6].str), (yyvsp[-5].str), (yyvsp[-3].str), (yyvsp[-2].kv), 1, NULL, NULL); }
#line 1540 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 47: /* create_corpus_llm_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_LLM T_STRING opt_llm_endpoint T_PROMPT T_STRING opt_with T_AUDIT_AGAINST T_IDENT opt_llm_audit_with  */
#line 205 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus_llm((yyvsp[-10].str), (yyvsp[-7].str), (yyvsp[-6].str), (yyvsp[-4].str), (yyvsp[-3].kv), 0, (yyvsp[-1].str), (yyvsp[0].kv)); }
#line 1546 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 48: /* create_corpus_llm_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_LLM T_STRING opt_llm_endpoint T_PROMPT T_STRING opt_with T_VERIFY_VIA T_PIPELINE_IR T_AUDIT_AGAINST T_IDENT opt_llm_audit_with  */
#line 210 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus_llm((yyvsp[-12].str), (yyvsp[-9].str), (yyvsp[-8].str), (yyvsp[-6].str), (yyvsp[-5].kv), 1, (yyvsp[-1].str), (yyvsp[0].kv)); }
#line 1552 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 49: /* opt_llm_endpoint: %empty  */
#line 213 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.str) = NULL; }
#line 1558 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 50: /* opt_llm_endpoint: T_AT T_STRING  */
#line 214 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.str) = (yyvsp[0].str); }
#line 1564 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 51: /* opt_llm_audit_with: %empty  */
#line 217 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = NULL; }
#line 1570 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 52: /* opt_llm_audit_with: T_WITH '(' kv_list ')'  */
#line 218 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = (yyvsp[-1].kv); }
#line 1576 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 53: /* create_corpus_oracle_stmt: T_CREATE T_CORPUS T_IDENT T_FROM T_ORACLE T_STRING opt_with opt_oracle_prompt  */
#line 229 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
      { (yyval.stmt) = oql_y_create_corpus_oracle((yyvsp[-5].str), (yyvsp[-2].str), (yyvsp[-1].kv), (yyvsp[0].str)); }
#line 1582 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 54: /* opt_oracle_prompt: %empty  */
#line 232 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.str) = NULL; }
#line 1588 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 55: /* opt_oracle_prompt: T_PROMPT T_STRING  */
#line 233 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.str) = (yyvsp[0].str); }
#line 1594 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 56: /* opt_with_bindings: %empty  */
#line 237 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = NULL; }
#line 1600 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 57: /* opt_with_bindings: T_WITH '(' binding_list ')'  */
#line 238 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = (yyvsp[-1].kv); }
#line 1606 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 58: /* binding_list: binding  */
#line 241 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = (yyvsp[0].kv); }
#line 1612 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 59: /* binding_list: binding_list ',' binding  */
#line 242 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = oql_y_kv_concat((yyvsp[-2].kv), (yyvsp[0].kv)); }
#line 1618 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 60: /* binding: T_IDENT T_EQ T_IDENT  */
#line 245 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                       { (yyval.kv) = oql_y_kv((yyvsp[-2].str), (yyvsp[0].str)); }
#line 1624 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 61: /* opt_with: %empty  */
#line 250 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.kv) = NULL; }
#line 1630 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 62: /* opt_with: T_WITH kv_list  */
#line 251 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.kv) = (yyvsp[0].kv); }
#line 1636 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 63: /* kv_list: kv  */
#line 254 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.kv) = (yyvsp[0].kv); }
#line 1642 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 64: /* kv_list: kv_list ',' kv  */
#line 255 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.kv) = oql_y_kv_concat((yyvsp[-2].kv), (yyvsp[0].kv)); }
#line 1648 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 65: /* kv: T_IDENT T_EQ value  */
#line 258 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.kv) = oql_y_kv((yyvsp[-2].str), (yyvsp[0].str)); }
#line 1654 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 66: /* value: T_STRING  */
#line 261 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
               { (yyval.str) = (yyvsp[0].str); }
#line 1660 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 67: /* value: T_NUMBER  */
#line 261 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                       { (yyval.str) = (yyvsp[0].str); }
#line 1666 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 68: /* value: T_IDENT  */
#line 261 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                              { (yyval.str) = (yyvsp[0].str); }
#line 1672 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 69: /* source: T_CORPUS T_STRING  */
#line 264 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { OqlSource s = {OQL_SRC_CORPUS, (yyvsp[0].str)}; (yyval.src) = s; }
#line 1678 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 70: /* source: T_STRING  */
#line 265 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { OqlSource s = {OQL_SRC_PATH,   (yyvsp[0].str)}; (yyval.src) = s; }
#line 1684 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 71: /* source: T_IDENT  */
#line 266 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { OqlSource s = {OQL_SRC_NAME,   (yyvsp[0].str)}; (yyval.src) = s; }
#line 1690 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 72: /* name_list: T_IDENT  */
#line 269 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.names) = oql_y_name((yyvsp[0].str)); }
#line 1696 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;

  case 73: /* name_list: name_list ',' T_IDENT  */
#line 270 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"
                                                      { (yyval.names) = oql_y_name_concat((yyvsp[-2].names), oql_y_name((yyvsp[0].str))); }
#line 1702 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"
    break;


#line 1706 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/build/microgpt_oql_parser.tab.c"

      default: break;
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
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == OQL_PARSER_EMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (parser, YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= OQL_PARSER_EOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == OQL_PARSER_EOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, parser);
          yychar = OQL_PARSER_EMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, parser);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (parser, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != OQL_PARSER_EMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, parser);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, parser);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 272 "/Users/user/dev/projects.github/microgpt-c/.claude/worktrees/agent-a6675de2724280a4e/src/microgpt_oql.y"

