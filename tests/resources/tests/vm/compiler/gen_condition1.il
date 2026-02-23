
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main(): boolean {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 4: if (amount < 50) amount = 0;
  4)	        CONDITION_LT	           amount:number     {n}	{c    }        50:number     {n}	{ r[0]}       r$0:boolean    {b}
  5)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L0 [8]

  6)	// *** Line 4: amount = 0;
  7)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}
  8)	L0:

  9)	// *** Line 6: if (amount < 100) {
 10)	        CONDITION_LT	           amount:number     {n}	{c    }       100:number     {n}	{ r[1]}       r$1:boolean    {b}
 11)	       JUMP_IF_FALSE	{ r[1]}       r$1:boolean    {b}	                 :              	                       L1 [15]

 12)	// *** Line 7: amount = 0;
 13)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}

 14)	// *** Line 10: if (amount >= 100) {
 15)	L1:

 16)	// *** Line 10: (amount >= 100) {
 17)	       CONDITION_GTE	           amount:number     {n}	{c    }       100:number     {n}	{ r[2]}       r$2:boolean    {b}
 18)	       JUMP_IF_FALSE	{ r[2]}       r$2:boolean    {b}	                 :              	                       L2 [22]

 19)	// *** Line 11: amount = 0;
 20)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}

 21)	// *** Line 14: var completed = amount != 0;
 22)	L2:

 23)	// *** Line 14: completed = amount != 0;
 24)	        CONDITION_NE	           amount:number     {n}	{c    }         0:number     {n}	{ r[3]}       r$3:boolean    {b}
 25)	      CREATE_SET_VAR	{ r[3]}       r$3:boolean    {b}	                 :              	        completed:boolean    {b}

 26)	// *** Line 16: return completed;
 27)	          STACK_PUSH	        completed:boolean    {b}	                 :              	                 :              
 28)	              RETURN	                 :              	                 :              	                 :              
