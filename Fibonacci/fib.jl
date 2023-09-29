function fib1(n)::BigInt 
if n <= 1
   n
else 
  fib1(n-1) fib1(n-2)
end
end
