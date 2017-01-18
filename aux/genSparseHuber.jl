# Generate data for sparse huber problem
# min || x ||₁  subj to  huber(b-Ax) ≤ σ.
#
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: January 10, 2017

function genSparseHuber(m, n, k, s, noise)
  # m,n size of problem
  # k = number of nonzeros in true solution
  # s = number of spikes/outliers in corrupted measurement b

  A = randn(m, n)

  # # Generate true solution (original method)
  # p = randperm(n)[1:k]
  # x0 = zeros(n)
  # x0[p] = randn(k)

  # Generate true solution: spike train.
  p = randperm(n)[1:k]
  x0 = zeros(n)
  x0[p] = sign(randn(k))

  # Generate RHS and corrupt.
  b = A*x0
  em = randn(m); em = em/norm(em)
  b = b + noise*em

  # Generate locations of spikes/outliers.
  sp = randperm(m)[1:s]
  e2 = randn(s); e2 = e2/norm(e2)
  b[sp] += 10*noise*e2

  outliers = zeros(m,1)
  outliers[sp] = 10*noise*e2

  σ = 2*noise

  return (A, b, x0, outliers, σ)

end
