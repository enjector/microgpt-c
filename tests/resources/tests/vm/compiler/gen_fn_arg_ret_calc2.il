
// ------------------------------------------------------------
// calc(amount, interest)
// ------------------------------------------------------------
  0)	calc:

  1)	// *** Line 1: function calc(amount: number, interest: number) : number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 1: interest: number) : number {
  4)	           STACK_POP	                 :              	                 :              	         interest:number     {n}

  5)	// *** Line 2: var k = amount * interest;
  6)	                 MUL	           amount:number     {n}	         interest:number     {n}	{ r[0]}       r$0:number     {n}
  7)	      CREATE_SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	                k:number     {n}

  8)	// *** Line 4: k = k + 1;
  9)	                 ADD	                k:number     {n}	{c    }         1:number     {n}	{ r[1]}       r$1:number     {n}
 10)	             SET_VAR	{ r[1]}       r$1:number     {n}	                 :              	                k:number     {n}

 11)	// *** Line 6: return k;
 12)	          STACK_PUSH	                k:number     {n}	                 :              	                 :              
 13)	              RETURN	                 :              	                 :              	                 :              
