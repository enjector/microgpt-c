
// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 2: function main() {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 4: if (amount < 100) {
  4)	        CONDITION_LT	           amount:number     {n}	{c    }       100:number     {n}	{ r[0]}       r$0:boolean    {b}
  5)	       JUMP_IF_FALSE	{ r[0]}       r$0:boolean    {b}	                 :              	                       L0 [10]

  6)	// *** Line 5: amount = 0;
  7)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}

  8)	// *** Line 7: else if (amount >= 100) {
  9)	                JUMP	                 :              	                 :              	                       L1 [26]
 10)	L0:

 11)	// *** Line 7: if (amount >= 100) {
 12)	       CONDITION_GTE	           amount:number     {n}	{c    }       100:number     {n}	{ r[1]}       r$1:boolean    {b}
 13)	       JUMP_IF_FALSE	{ r[1]}       r$1:boolean    {b}	                 :              	                       L2 [18]

 14)	// *** Line 8: amount = 0;
 15)	             SET_VAR	{c    }         0:number     {n}	                 :              	           amount:number     {n}

 16)	// *** Line 9: else if (amount > 1000) {
 17)	                JUMP	                 :              	                 :              	                       L3 [25]
 18)	L2:

 19)	// *** Line 9: if (amount > 1000) {
 20)	        CONDITION_GT	           amount:number     {n}	{c    }      1000:number     {n}	{ r[2]}       r$2:boolean    {b}
 21)	       JUMP_IF_FALSE	{ r[2]}       r$2:boolean    {b}	                 :              	                       L4 [24]

 22)	// *** Line 10: amount = 1000;
 23)	             SET_VAR	{c    }      1000:number     {n}	                 :              	           amount:number     {n}
 24)	L4:
 25)	L3:
 26)	L1:
 27)	              RETURN	                 :              	                 :              	                 :              
