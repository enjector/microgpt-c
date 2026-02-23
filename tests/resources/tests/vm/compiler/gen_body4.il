
// ------------------------------------------------------------
// debug(value)
// ------------------------------------------------------------
  0)	debug:

  1)	// *** Line 3: /*
  2)	     CALL_EXT_METHOD	                 :              	                 :              	                 :              
  3)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// calculate_interest(amount, rate, max)
// ------------------------------------------------------------
  0)	calculate_interest:

  1)	// *** Line 5: function calculate_interest(amount: number, rate: number, max: number): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 5: rate: number, max: number): number {
  4)	           STACK_POP	                 :              	                 :              	             rate:number     {n}

  5)	// *** Line 5: max: number): number {
  6)	           STACK_POP	                 :              	                 :              	              max:number     {n}

  7)	// *** Line 6: var total = amount * rate;
  8)	                 MUL	           amount:number     {n}	             rate:number     {n}	{ r[0]}       r$0:number     {n}
  9)	      CREATE_SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	            total:number     {n}

 10)	// *** Line 8: debug(max);
 11)	          STACK_PUSH	              max:number     {n}	                 :              	                 :              
 12)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

 13)	// *** Line 10: if (!max) max = 100.1;
 14)	                 NOT	              max:number     {n}	                 :              	{ r[1]}       r$1:boolean    {b}
 15)	      CONDITION_TRUE	{ r[1]}       r$1:boolean    {b}	                 :              	{ r[2]}       r$2:boolean    {b}
 16)	       JUMP_IF_FALSE	{ r[2]}       r$2:boolean    {b}	                 :              	                       L0 [19]

 17)	// *** Line 10: max = 100.1;
 18)	             SET_VAR	{c    }     100.1:number     {n}	                 :              	              max:number     {n}
 19)	L0:

 20)	// *** Line 12: if (total > max) {
 21)	        CONDITION_GT	            total:number     {n}	              max:number     {n}	{ r[3]}       r$3:boolean    {b}
 22)	       JUMP_IF_FALSE	{ r[3]}       r$3:boolean    {b}	                 :              	                       L1 [26]

 23)	// *** Line 13: total = max;
 24)	             SET_VAR	              max:number     {n}	                 :              	            total:number     {n}

 25)	// *** Line 16: debug(total);
 26)	L1:

 27)	// *** Line 16: total);
 28)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 29)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

 30)	// *** Line 18: return total;
 31)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 32)	              RETURN	                 :              	                 :              	                 :              

 33)	// *** Line 19: /*
 * test.ts
 * Some useful info
 */
 34)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 23: function main(): number {
  2)	      CREATE_SET_VAR	{c    }     10.12:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 24: /*
 * test.ts
 * Some useful info
 */

  4)	// *** Line 26: // Do calculation

  5)	// *** Line 27: var total = calculate_interest(amount, 1.12345);
  6)	          STACK_PUSH	{c    }   1.12345:number     {n}	                 :              	                 :              
  7)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
  8)	         CALL_METHOD	       calculate_interest:number     {n}	                 :              	                 :              
  9)	           STACK_POP	                 :              	                 :              	{ r[0]}       r$0:number     {n}
 10)	      CREATE_SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	            total:number     {n}

 11)	// *** Line 28: debug(total);
 12)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 13)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

 14)	// *** Line 30: total = calculate_interest(amount, 1.12345, 200.12);
 15)	          STACK_PUSH	{c    }    200.12:number     {n}	                 :              	                 :              
 16)	          STACK_PUSH	{c    }   1.12345:number     {n}	                 :              	                 :              
 17)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 18)	         CALL_METHOD	       calculate_interest:number     {n}	                 :              	                 :              
 19)	           STACK_POP	                 :              	                 :              	{ r[1]}       r$1:number     {n}
 20)	             SET_VAR	{ r[1]}       r$1:number     {n}	                 :              	            total:number     {n}

 21)	// *** Line 32: debug(total);
 22)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 23)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

 24)	// *** Line 34: // Finished

 25)	// *** Line 36: return total;
 26)	          STACK_PUSH	            total:number     {n}	                 :              	                 :              
 27)	              RETURN	                 :              	                 :              	                 :              

 28)	// *** Line 39: /*
 * test.ts
 * Some useful info
 */

// ** ERRORS ******************************************************************
- Line 16: Incorrect parameter arg 1 (value) to fn debug in calculate_interest: expected string but got number
- Line 8: Incorrect parameter arg 1 (value) to fn debug in calculate_interest: expected string but got number
- Line 32: Incorrect parameter arg 1 (value) to fn debug in main: expected string but got number
- Line 28: Incorrect parameter arg 1 (value) to fn debug in main: expected string but got number
- Line 27: Incorrect number of arguments to fn calculate_interest: expected 3, but got 2
