/*
 * microgpt_oql.l  —  Flex lexer for OQL (Organelle Query Language).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Tokenises a SQL-shaped DSL with a hard-locked +6 / -4 verb surface.
 * Generated symbols are prefixed "oql_parser_" via %option prefix to
 * coexist with the VM's Flex/Bison.
 *
 * Input is in-memory (custom YY_INPUT); see microgpt_oql.c for the driver.
 */

%option prefix="oql_parser_"
%option noyywrap
%option nounistd

%{
    #include "microgpt_oql.h"
    #include "microgpt_oql_parser.tab.h"
    #include <stdlib.h>
    #include <string.h>

    /* isatty / fileno declarations — Flex's generated scanner uses them. */
    #ifndef _WIN32
    #include <unistd.h>
    #else
    #include <io.h>
    #define isatty _isatty
    #endif

    extern oql_parser *_oql_current_parser;

    static char *oql_strclone(const char *s) {
        size_t n = strlen(s) + 1;
        char *d = (char *)malloc(n);
        if (d) memcpy(d, s, n);
        return d;
    }
    /* Strip surrounding single quotes from a STRING lexeme, returning a new
     * heap-allocated copy (caller owns). */
    static char *oql_str_unquote(const char *s) {
        size_t n = strlen(s);
        if (n >= 2 && s[0] == '\'' && s[n - 1] == '\'') {
            char *d = (char *)malloc(n - 1);
            if (!d) return NULL;
            memcpy(d, s + 1, n - 2);
            d[n - 2] = '\0';
            return d;
        }
        return oql_strclone(s);
    }

    #undef YY_INPUT
    #define YY_INPUT(buf, result, max_size) \
    { \
        int c = oql_parser_char_fetch_next(_oql_current_parser); \
        result = (c == 0) ? YY_NULL : (buf[0] = c, 1); \
    }

    #define YY_DECL int oql_parser_lex(oql_parser *parser)
%}

ws      [ \t]+
nl      (\r\n|\r|\n)
ident   [A-Za-z_][A-Za-z0-9_]*
number  -?[0-9]+(\.[0-9]+)?
string  \'([^'\\]|\\.)*\'
comment --[^\n\r]*
graph   @graph[^@]*@end
vmbody  `[^`]*`

%%

 /* Verbs (the +6 — hard-locked).  CREATE is inherited from SQL — not a
  * 7th verb; subtype carried by following object-type keyword. */
TRAIN       { return T_TRAIN; }
COMPOSE     { return T_COMPOSE; }
RUN         { return T_RUN; }
EVALUATE    { return T_EVALUATE; }
VERIFY      { return T_VERIFY; }
AUDIT       { return T_AUDIT; }
CREATE      { return T_CREATE; }

 /* Object-type keywords (for CREATE). */
BEHAVIOUR   { return T_BEHAVIOUR; }
ORGANELLE   { return T_ORGANELLE; }
CHECKPOINT  { return T_CHECKPOINT; }
VM          { return T_VM; }
 /* E10 — CORPUS already exists below as a source-kind keyword; FILE is new. */
FILE        { return T_FILE; }

 /* Clauses / sub-keywords. */
ON          { return T_ON; }
WITH        { return T_WITH; }
AGAINST     { return T_AGAINST; }
USING       { return T_USING; }
WHERE       { return T_WHERE; }
AS          { return T_AS; }
FROM        { return T_FROM; }
GRAPH       { return T_GRAPH; }
CORPUS      { return T_CORPUS; }
REPORT      { return T_REPORT; }
METRIC      { return T_METRIC; }
THRESHOLDS  { return T_THRESHOLDS; }

 /* Inline @graph...@end as a single opaque token. */
{graph}     { oql_parser_lval.str = oql_strclone(yytext); return T_GRAPH_BLOCK; }
 /* Backtick-delimited VM body (TS source) — stripped of the delimiters. */
{vmbody}    {
                /* yytext includes the bracketing backticks; strip them. */
                size_t n = strlen(yytext);
                char *body = (char *)malloc(n - 1);
                if (body) {
                    memcpy(body, yytext + 1, n - 2);
                    body[n - 2] = '\0';
                }
                oql_parser_lval.str = body;
                return T_VM_BODY;
            }

{string}    { oql_parser_lval.str = oql_str_unquote(yytext); return T_STRING; }
{number}    { oql_parser_lval.str = oql_strclone(yytext); return T_NUMBER; }
{ident}     { oql_parser_lval.str = oql_strclone(yytext); return T_IDENT; }

"<="        { return T_LE; }
">="        { return T_GE; }
"!="        { return T_NE; }
"="         { return T_EQ; }
"<"         { return T_LT; }
">"         { return T_GT; }

{comment}   { /* skip line comment */ }
{ws}        { /* skip whitespace */ }
{nl}        { parser->line_number++; }

.           { return yytext[0]; }

%%
