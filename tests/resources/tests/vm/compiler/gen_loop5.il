
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main() {
  2)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	            count:number     {n}

  3)	// *** Line 3: for (var i = 0; i < 10; i++) {
  4)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	                i:number     {n}

  5)	// *** Line 3: for_loop_begin
  6)	L0:

  7)	// *** Line 3: i < 10; i++) {
  8)	        CONDITION_LT	                i:number     {n}	{c    }        10:number     {n}	{ r[0]}       r$0:boolean    {b}
  9)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L1 [17]

 10)	// *** Line 4: ) {
 11)	                 INC	            count:number     {n}	                 :              	                 :              

 12)	// *** Line 3: for_increment_expression

 13)	// *** Line 3: i++) {
 14)	                 INC	                i:number     {n}	                 :              	                 :              

 15)	// *** Line 5: for_loop_end
 16)	                JUMP	                 :              	                 :              	                       L0 [6]
 17)	L1:

 18)	// *** Line 7: var total = 0;
 19)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	            total:number     {n}

 20)	// *** Line 8: for (var j = 0; j < 10; j++) {
 21)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	                j:number     {n}

 22)	// *** Line 8: for_loop_begin
 23)	L2:

 24)	// *** Line 8: j < 10; j++) {
 25)	        CONDITION_LT	                j:number     {n}	{c    }        10:number     {n}	{ r[1]}       r$1:boolean    {b}
 26)	       JUMP_IF_FALSE	{ r[1]}       r$1:boolean    {b}	                 :              	                       L3 [34]

 27)	// *** Line 9: ) {
 28)	                 ADD	            total:number     {n}	                j:number     {n}	            total:number     {n}

 29)	// *** Line 8: for_increment_expression

 30)	// *** Line 8: j++) {
 31)	                 INC	                j:number     {n}	                 :              	                 :              

 32)	// *** Line 10: for_loop_end
 33)	                JUMP	                 :              	                 :              	                       L2 [23]
 34)	L3:
 35)	              RETURN	                 :              	                 :              	                 :              
