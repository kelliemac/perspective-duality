# Using Chambolle-Pock, solve the robust regression primal problem
# min     ||x||₁
# s.t.    huber(b-Ax) ≤ σ.
#
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: January 10, 2017

using Convex; cvx = Convex
#include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/algorithms/cp.jl")
#include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/aux/proj_infball.jl")

# Define projection of w onto Huber constraint set in proxFc.
# That is, given y solve
# min    norm(y-z)^2
# s.t.     ρ(b-z) ≤ σ
# where ρ is Huber function with parameter η.
function proj_huber(y, b, σ, η; tol = 1e-9, kwargs...)
  m = size(y,1)

  z = cvx.Variable(m)
  w = cvx.Variable(m)
  s = cvx.Variable(m)
  p = cvx.minimize( sumsquares(y-z) )

  for i in 1:m
    p.constraints += (0.5/η)*huber( w[i], η ) ≤ s[i]
  end
  p.constraints += (sum(s) ≤ σ)
  p.constraints += (w == b-z)

  solve!(p, GurobiSolver(OutputFlag=0, OptimalityTol=tol) )
  zproj= z.value[:,1]

  return zproj

end


# Run Chambolle-Pock to solve robust regression primal problem.
function inf_hubPrimal(A, b, σ, η; kwargs... )
  (m, n) = size(A)

  # Provide prox of G = ||⋅||₁
  function proxG(x, t)
    prox = x - proj_infball(x, t)
    return prox
  end

  # Provide prox of F*, where F(y) = δ_{ρ(b - ⋅) ≤ σ}(y)
  # and ρ is the Huber with parameter η.
  function proxFc(y, t)
    proj = proj_huber(y/t, b, σ, η; kwargs...)
    prox = y - t*proj # use moreau identity
    return prox
  end

  # Initialize CP
  x0 = ones(n)/n
  y0 = ones(m)/m

  (iterates, steps, xfinal) = chamb_pock( proxFc, proxG, A, x0, y0; kwargs...)
  return (iterates, steps, xfinal)

end
