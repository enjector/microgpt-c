
// ------------------------------------------------------------
// calculate_interest3(amount, rate)
// ------------------------------------------------------------
  0)	calculate_interest3:

  1)	// *** Line 1: function calculate_interest3(amount: number, rate: number): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 1: rate: number): number {
  4)	           STACK_POP	                 :              	                 :              	             rate:number     {n}

  5)	// *** Line 2: return amount * rate;
  6)	                 MUL	           amount:number     {n}	             rate:number     {n}	{ r[0]}       r$0:number     {n}
  7)	          STACK_PUSH	{ r[0]}       r$0:number     {n}	                 :              	                 :              
  8)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// add_bonus(amount)
// ------------------------------------------------------------
  0)	add_bonus:

  1)	// *** Line 5: function add_bonus(amount: number): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 6: return amount + 100;
  4)	                 ADD	           amount:number     {n}	{c    }       100:number     {n}	{ r[0]}       r$0:number     {n}
  5)	          STACK_PUSH	{ r[0]}       r$0:number     {n}	                 :              	                 :              
  6)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 10: function main() {
  2)	      CREATE_SET_VAR	{c    }        10:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 12: // test

  4)	// *** Line 13: amount = amount + 120;
  5)	                 ADD	           amount:number     {n}	{c    }       120:number     {n}	{ r[0]}       r$0:number     {n}
  6)	             SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	           amount:number     {n}

  7)	// *** Line 15: // Do calculation

  8)	// *** Line 16: var total = add_bonus(calculate_interest3(amount, 1.12345));
  9)	          STACK_PUSH	{c    }   1.12345:number     {n}	                 :              	                 :              
 10)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 11)	         CALL_METHOD	       calculate_interest3:number     {n}	                 :              	                 :              
 12)	           STACK_POP	                 :              	                 :              	{ r[1]}       r$1:number     {n}

 13)	// *** Line 16: );
 14)	          STACK_PUSH	{ r[1]}       r$1:number     {n}	                 :              	                 :              
 15)	         CALL_METHOD	        add_bonus:number     {n}	                 :              	                 :              
 16)	           STACK_POP	                 :              	                 :              	{ r[2]}       r$2:number     {n}
 17)	      CREATE_SET_VAR	{ r[2]}       r$2:number     {n}	                 :              	            total:number     {n}

 18)	// *** Line 17: //debug(total);

 19)	// *** Line 19: // Finished
 20)	              RETURN	                 :              	                 :              	                 :              
