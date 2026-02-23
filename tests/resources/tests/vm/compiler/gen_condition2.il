
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main() {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 4: if (amount < 10) amount = 0;
  4)	        CONDITION_LT	           amount:number     {n}	{c    }        10:number     {n}	{ r[0]}       r$0:boolean    {b}
  5)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L0 [8]

  6)	// *** Line 4: amount = 0;
  7)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}
  8)	L0:

  9)	// *** Line 6: if (amount < 100) {
 10)	        CONDITION_LT	           amount:number     {n}	{c    }       100:number     {n}	{ r[1]}       r$1:boolean    {b}
 11)	       JUMP_IF_FALSE	{ r[1]}       r$1:boolean    {b}	                 :              	                       L1 [16]

 12)	// *** Line 7: amount = 0;
 13)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}

 14)	// *** Line 8: else {
 15)	                JUMP	                 :              	                 :              	                       L2 [19]
 16)	L1:

 17)	// *** Line 9: amount = 1000;
 18)	             SET_VAR	{c    }      1000:number     {n}	                 :              	           amount:number     {n}
 19)	L2:
 20)	              RETURN	                 :              	                 :              	                 :              
