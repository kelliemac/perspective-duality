# Using Convex (CVX), solve the robust regression primal problem
# min     ||x||₁
# s.t.    huber(b-Ax) ≤ σ.
#
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: January 10, 2017

using Convex; cvx = Convex

function inf_hubPrimal_cvx(A, b, σ, η)

  (m,n) = size(A)

  x = cvx.Variable(n)
  s = cvx.Variable(m)
  t = cvx.Variable(m)

  p = cvx.minimize( norm(x,1) )

  for i in 1:m
    p.constraints += (0.5/η)*huber(s[i], η) ≤ t[i]
  end
  p.constraints += sum(t) ≤ σ
  p.constraints += b - A*x == s

  solve!(p, GurobiSolver())

  xprimal = x.value[:,1]

  return xprimal

end
