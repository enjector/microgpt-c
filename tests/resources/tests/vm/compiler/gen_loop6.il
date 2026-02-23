
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main() {
  2)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	            total:number     {n}

  3)	// *** Line 3: for (var i = 0; i < 10; i++) {
  4)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	                i:number     {n}

  5)	// *** Line 3: for_loop_begin
  6)	L0:

  7)	// *** Line 3: i < 10; i++) {
  8)	        CONDITION_LT	                i:number     {n}	{c    }        10:number     {n}	{ r[0]}       r$0:boolean    {b}
  9)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L1 [31]

 10)	// *** Line 4: ) {
 11)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	                j:number     {n}

 12)	// *** Line 4: for_loop_begin
 13)	L2:

 14)	// *** Line 4: j < 10; j++) {
 15)	        CONDITION_LT	                j:number     {n}	{c    }        10:number     {n}	{ r[1]}       r$1:boolean    {b}
 16)	       JUMP_IF_FALSE	{ r[1]}       r$1:boolean    {b}	                 :              	                       L3 [25]

 17)	// *** Line 5: ) {
 18)	                 MUL	                i:number     {n}	                j:number     {n}	{ r[2]}       r$2:number     {n}
 19)	                 ADD	            total:number     {n}	{ r[2]}       r$2:number     {n}	            total:number     {n}

 20)	// *** Line 4: for_increment_expression

 21)	// *** Line 4: j++) {
 22)	                 INC	                j:number     {n}	                 :              	                 :              

 23)	// *** Line 6: for_loop_end
 24)	                JUMP	                 :              	                 :              	                       L2 [13]
 25)	L3:

 26)	// *** Line 3: for_increment_expression

 27)	// *** Line 3: i++) {
 28)	                 INC	                i:number     {n}	                 :              	                 :              

 29)	// *** Line 7: for_loop_end
 30)	                JUMP	                 :              	                 :              	                       L0 [6]
 31)	L1:
 32)	              RETURN	                 :              	                 :              	                 :              
