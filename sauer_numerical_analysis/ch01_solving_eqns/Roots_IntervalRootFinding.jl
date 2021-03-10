using Roots

f(x) = x^3 + x - 1

find_zero(f, 0, 1, Roots.Brent()) # mimics fzero from matlab


using IntervalArithmetic, IntervalRootFinding

roots(f, 0..1)

