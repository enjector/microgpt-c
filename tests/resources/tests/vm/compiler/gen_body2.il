
// ------------------------------------------------------------
// debug(value)
// ------------------------------------------------------------
  0)	debug:

  1)	// *** Line 1: declare function debug(value: number) : void;
  2)	     CALL_EXT_METHOD	                 :              	                 :              	                 :              
  3)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// get_rate(country)
// ------------------------------------------------------------
  0)	get_rate:

  1)	// *** Line 3: function get_rate(country: string): number {
  2)	           STACK_POP	                 :              	                 :              	          country:string     {s}

  3)	// *** Line 4: return 1.34;
  4)	          STACK_PUSH	{c    }      1.34:number     {n}	                 :              	                 :              
  5)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// calculate_interest2(amount, country)
// ------------------------------------------------------------
  0)	calculate_interest2:

  1)	// *** Line 7: function calculate_interest2(amount: number, country: string): number {
  2)	           STACK_POP	                 :              	                 :              	           amount:number     {n}

  3)	// *** Line 7: country: string): number {
  4)	           STACK_POP	                 :              	                 :              	          country:string     {s}

  5)	// *** Line 8: return amount * get_rate() * global_rate;
  6)	         CALL_METHOD	         get_rate:number     {n}	                 :              	                 :              
  7)	           STACK_POP	                 :              	                 :              	{ r[0]}       r$0:number     {n}

  8)	// *** Line 8: * global_rate;
  9)	                 MUL	           amount:number     {n}	{ r[0]}       r$0:number     {n}	{ r[1]}       r$1:number     {n}

 10)	// *** Line 8: global_rate;
 11)	                 MUL	{ r[1]}       r$1:number     {n}	       global_rate:?             	{ r[2]}       r$2:number     {n}
 12)	          STACK_PUSH	{ r[2]}       r$2:number     {n}	                 :              	                 :              
 13)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// dosomething(name)
// ------------------------------------------------------------
  0)	dosomething:

  1)	// *** Line 11: function dosomething(name: string) {
  2)	           STACK_POP	                 :              	                 :              	             name:string     {s}

  3)	// *** Line 12: var a = 1;
  4)	      CREATE_SET_VAR	{c    }         1:number     {n}	                 :              	                a:number     {n}
  5)	              RETURN	                 :              	                 :              	                 :              

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
  0)	main:

  1)	// *** Line 16: function main() {
  2)	      CREATE_SET_VAR	{c    }       100:number     {n}	                 :              	           amount:number     {n}

  3)	// *** Line 18: amount = amount + 100;
  4)	                 ADD	           amount:number     {n}	{c    }       100:number     {n}	{ r[0]}       r$0:number     {n}
  5)	             SET_VAR	{ r[0]}       r$0:number     {n}	                 :              	           amount:number     {n}

  6)	// *** Line 20: var interest = calculate_interest2(amount, "UK");
  7)	          STACK_PUSH	{c    }        UK:string     {s}	                 :              	                 :              
  8)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
  9)	         CALL_METHOD	       calculate_interest2:number     {n}	                 :              	                 :              
 10)	           STACK_POP	                 :              	                 :              	{ r[1]}       r$1:number     {n}
 11)	      CREATE_SET_VAR	{ r[1]}       r$1:number     {n}	                 :              	         interest:number     {n}

 12)	// *** Line 21: debug(interest);
 13)	          STACK_PUSH	         interest:number     {n}	                 :              	                 :              
 14)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              

 15)	// *** Line 23: interest = calculate_interest2("USA", amount);
 16)	          STACK_PUSH	           amount:number     {n}	                 :              	                 :              
 17)	          STACK_PUSH	{c    }       USA:string     {s}	                 :              	                 :              
 18)	         CALL_METHOD	       calculate_interest2:number     {n}	                 :              	                 :              
 19)	           STACK_POP	                 :              	                 :              	{ r[2]}       r$2:number     {n}
 20)	             SET_VAR	{ r[2]}       r$2:number     {n}	                 :              	         interest:number     {n}

 21)	// *** Line 24: debug(interest);
 22)	          STACK_PUSH	         interest:number     {n}	                 :              	                 :              
 23)	         CALL_METHOD	            debug:void       {o}	                 :              	                 :              
 24)	              RETURN	                 :              	                 :              	                 :              

// ** ERRORS ******************************************************************
- Line 8: Incorrect number of arguments to fn get_rate: expected 1, but got 0
- Line 23: Incorrect parameter arg 1 (amount) to fn calculate_interest2 in main: expected number but got string
- Line 8: Undeclared variable global_rate used in function calculate_interest2(...)
