
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main(): number {
  2)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	            total:number     {n}

  3)	// *** Line 4: for (var i = 0; i < 10; i++) {
  4)	      CREATE_SET_VAR	{c    }         0:number     {n}	                 :              	                i:number     {n}

  5)	// *** Line 4: for_loop_begin
  6)	L0:

  7)	// *** Line 4: i < 10; i++) {
  8)	        CONDITION_LT	                i:number     {n}	{c    }        10:number     {n}	{ r[0]}       r$0:boolean    {b}
  9)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L1 [17]

 10)	// *** Line 5: ) {
 11)	                 ADD	            total:number     {n}	                i:number     {n}	            total:number     {n}

 12)	// *** Line 4: for_increment_expression

 13)	// *** Line 4: i++) {
 14)	                 INC	                i:number     {n}	                 :              	                 :              

 15)	// *** Line 6: for_loop_end
 16)	                JUMP	                 :              	                 :              	                       L0 [6]
 17)	L1:

 18)	// *** Line 8: return total;
 19)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 20)	              RETURN	                 :              	                 :              	                 :              
