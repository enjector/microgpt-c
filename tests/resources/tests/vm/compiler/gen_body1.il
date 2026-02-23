
// ------------------------------------------------------------
// debug(value)
// ------------------------------------------------------------
  0)	debug:

  1)	// *** Line 1: declare function debug(value: number) : void;
  2)	     CALL_EXT_METHOD	                 :              	                 :              	                 :              
  3)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 4: function main(): number {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 6: amount = amount + 100;
  4)	                 ADD	           amount:number     {n}	{c    }       100:number     {n}	{ r[0]}       r$0:number     {n}
  5)	             SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	           amount:number     {n}

  6)	// *** Line 8: debug(amount);
  7)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
  8)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

  9)	// *** Line 10: return amount;
 10)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 11)	              RETURN	                 :              	                 :              	                 :              
