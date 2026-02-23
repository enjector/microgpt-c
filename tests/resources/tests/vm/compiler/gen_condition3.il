
// ------------------------------------------------------------
// test_a(amount)
// ------------------------------------------------------------
  0)	test_a:

  1)	// *** Line 1: function test_a(amount: number): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 2: var reset = false;
  4)	      CREATE_SET_VAR	{c    }     false:boolean    {b}	                 :              	            reset:boolean    {b}

  5)	// *** Line 3: amount = amount + 1;
  6)	                 ADD	           amount:number     {n}	{c    }         1:number     {n}	{ r[0]}       r$0:number     {n}
  7)	             SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	           amount:number     {n}

  8)	// *** Line 5: if (amount < 100 || reset) {
  9)	        CONDITION_LT	           amount:number     {n}	{c    }       100:number     {n}	{ r[1]}       r$1:boolean    {b}

 10)	// *** Line 5: reset) {
 11)	      CONDITION_TRUE	            reset:boolean    {b}	                 :              	{ r[2]}       r$2:boolean    {b}
 12)	      CONDITIONAL_OR	{ r[1]}       r$1:boolean    {b}	{ r[2]}       r$2:boolean    {b}	{ r[3]}       r$3:boolean    {b}
 13)	       JUMP_IF_FALSE	{ r[3]}       r$3:boolean    {b}	                 :              	                       L0 [18]

 14)	// *** Line 6: return 0;
 15)	          STACK_PUSH	{c    }         0:number     {n}	                 :              	                 :              
 16)	              RETURN	                 :              	                 :              	                 :              

 17)	// *** Line 9: return amount;
 18)	L0:

 19)	// *** Line 9: amount;
 20)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 21)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// test_b(amount)
// ------------------------------------------------------------
  0)	test_b:

  1)	// *** Line 12: function test_b(amount: number): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 13: var check = amount;
  4)	      CREATE_SET_VAR	           amount:number     {n}	                 :              	            check:number     {n}

  5)	// *** Line 14: if (amount >= 100 || check > 9999) {
  6)	       CONDITION_GTE	           amount:number     {n}	{c    }       100:number     {n}	{ r[0]}       r$0:boolean    {b}

  7)	// *** Line 14: check > 9999) {
  8)	        CONDITION_GT	            check:number     {n}	{c    }      9999:number     {n}	{ r[1]}       r$1:boolean    {b}
  9)	      CONDITIONAL_OR	{ r[0]}       r$0:boolean    {b}	{ r[1]}       r$1:boolean    {b}	{ r[2]}       r$2:boolean    {b}
 10)	       JUMP_IF_FALSE	{ r[2]}       r$2:boolean    {b}	                 :              	                       L0 [16]

 11)	// *** Line 15: return 0;
 12)	          STACK_PUSH	{c    }         0:number     {n}	                 :              	                 :              
 13)	              RETURN	                 :              	                 :              	                 :              

 14)	// *** Line 16: else {
 15)	                JUMP	                 :              	                 :              	                       L1 [20]
 16)	L0:

 17)	// *** Line 17: return 10;
 18)	          STACK_PUSH	{c    }        10:number     {n}	                 :              	                 :              
 19)	              RETURN	                 :              	                 :              	                 :              
 20)	L1:
 21)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 22: function main(): number {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 24: test_a(amount);
  4)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
  5)	         CALL_METHOD	           test_a:number     {n}	                 :              	                 :              

  6)	// *** Line 25: test_b(amount);
  7)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
  8)	         CALL_METHOD	           test_b:number     {n}	                 :              	                 :              

  9)	// *** Line 27: return amount;
 10)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 11)	              RETURN	                 :              	                 :              	                 :              
