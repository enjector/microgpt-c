/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_OQL_PARSER_MICROGPT_OQL_PARSER_TAB_H_INCLUDED
# define YY_OQL_PARSER_MICROGPT_OQL_PARSER_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef OQL_PARSER_DEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define OQL_PARSER_DEBUG 1
#  else
#   define OQL_PARSER_DEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define OQL_PARSER_DEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined OQL_PARSER_DEBUG */
#if OQL_PARSER_DEBUG
extern int oql_parser_debug;
#endif

/* Token kinds.  */
#ifndef OQL_PARSER_TOKENTYPE
# define OQL_PARSER_TOKENTYPE
  enum oql_parser_tokentype
  {
    OQL_PARSER_EMPTY = -2,
    OQL_PARSER_EOF = 0,            /* "end of file"  */
    OQL_PARSER_error = 256,        /* error  */
    OQL_PARSER_UNDEF = 257,        /* "invalid token"  */
    T_TRAIN = 258,                 /* T_TRAIN  */
    T_COMPOSE = 259,               /* T_COMPOSE  */
    T_RUN = 260,                   /* T_RUN  */
    T_EVALUATE = 261,              /* T_EVALUATE  */
    T_VERIFY = 262,                /* T_VERIFY  */
    T_AUDIT = 263,                 /* T_AUDIT  */
    T_ON = 264,                    /* T_ON  */
    T_WITH = 265,                  /* T_WITH  */
    T_AGAINST = 266,               /* T_AGAINST  */
    T_USING = 267,                 /* T_USING  */
    T_WHERE = 268,                 /* T_WHERE  */
    T_AS = 269,                    /* T_AS  */
    T_FROM = 270,                  /* T_FROM  */
    T_GRAPH = 271,                 /* T_GRAPH  */
    T_CORPUS = 272,                /* T_CORPUS  */
    T_REPORT = 273,                /* T_REPORT  */
    T_METRIC = 274,                /* T_METRIC  */
    T_THRESHOLDS = 275,            /* T_THRESHOLDS  */
    T_LT = 276,                    /* T_LT  */
    T_LE = 277,                    /* T_LE  */
    T_EQ = 278,                    /* T_EQ  */
    T_NE = 279,                    /* T_NE  */
    T_GE = 280,                    /* T_GE  */
    T_GT = 281,                    /* T_GT  */
    T_IDENT = 282,                 /* T_IDENT  */
    T_STRING = 283,                /* T_STRING  */
    T_NUMBER = 284,                /* T_NUMBER  */
    T_GRAPH_BLOCK = 285            /* T_GRAPH_BLOCK  */
  };
  typedef enum oql_parser_tokentype oql_parser_token_kind_t;
#endif

/* Value type.  */
#if ! defined OQL_PARSER_STYPE && ! defined OQL_PARSER_STYPE_IS_DECLARED
union OQL_PARSER_STYPE
{
#line 40 "microgpt_oql.y"

    char *str;
    int   op;
    OqlKV *kv;
    OqlNameList *names;
    OqlSource src;
    OqlPredicate *pred;
    OqlStmt *stmt;

#line 112 "microgpt_oql_parser.tab.h"

};
typedef union OQL_PARSER_STYPE OQL_PARSER_STYPE;
# define OQL_PARSER_STYPE_IS_TRIVIAL 1
# define OQL_PARSER_STYPE_IS_DECLARED 1
#endif


extern OQL_PARSER_STYPE oql_parser_lval;


int oql_parser_parse (oql_parser *parser);


#endif /* !YY_OQL_PARSER_MICROGPT_OQL_PARSER_TAB_H_INCLUDED  */
