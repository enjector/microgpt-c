
// ------------------------------------------------------------
// wrong_return_type(amount)
// ------------------------------------------------------------
  0)	wrong_return_type:

  1)	// *** Line 9: // Test test.js
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 10: return 99.133;
  4)	          STACK_PUSH	{c    }    99.133:number     {n}	                 :              	                 :              
  5)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// wrong_return_type2()
// ------------------------------------------------------------
  0)	wrong_return_type2:

  1)	// *** Line 16: /* This is a test
that wraps
multiple lines */

  2)	// *** Line 17: var var1 = true;
  3)	      CREATE_SET_VAR	{c    }      true:boolean    {b}	                 :              	             var1:boolean    {b}

  4)	// *** Line 19: return var1;
  5)	          STACK_PUSH	             var1:boolean    {b}	                 :              	                 :              
  6)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// my_function()
// ------------------------------------------------------------
  0)	my_function:

  1)	// *** Line 23: function my_function(): number {
  2)	          STACK_PUSH	{c    }       123:number     {n}	                 :              	                 :              
  3)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 27: function main() {
  2)	      CREATE_SET_VAR	{c    }     10.12:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 29: // Wrong assignment

  4)	// *** Line 30: amount = wrong_return_type(1234.1);
  5)	          STACK_PUSH	{c    }    1234.1:number     {n}	                 :              	                 :              
  6)	         CALL_METHOD	       wrong_return_type:boolean    {b}	                 :              	                 :              
  7)	           STACK_POP	                 :              	                 :              	{ r[0]}       r$0:boolean    {b}
  8)	             SET_VAR	{ r[0]}       r$0:boolean    {b}	                 :              	           amount:number     {n}

  9)	// *** Line 32: var z = true;
 10)	      CREATE_SET_VAR	{c    }      true:boolean    {b}	                 :              	                z:boolean    {b}

 11)	// *** Line 34: z = my_function();
 12)	         CALL_METHOD	       my_function:number     {n}	                 :              	                 :              
 13)	           STACK_POP	                 :              	                 :              	{ r[1]}       r$1:number     {n}
 14)	             SET_VAR	{ r[1]}       r$1:number     {n}	                 :              	                z:boolean    {b}
 15)	              RETURN	                 :              	                 :              	                 :              

 16)	// *** Line 37: // Comment 1b

 17)	// *** Line 39: /*
 * test.ts
 * Some useful info
 */

// ** ERRORS ******************************************************************
- Line 10: Wrong type 99.133:number being returned from function wrong_return_type(...):boolean
- Line 19: Wrong type var1:boolean being returned from function wrong_return_type2(...):number
- Line 30: Return of function wrong_return_type(...):boolean cannot be assigned to variable amount:number
- Line 34: Return of function my_function(...):number cannot be assigned to variable z:boolean
