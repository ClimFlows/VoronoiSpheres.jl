# generic

apply_primal(op, sphere, arg::Vector, more...) = [
    op(op(sphere), cell)(arg, more...) for cell in cells(sphere)
]
apply_primal(op, sphere, arg::Matrix, more...) = [
    op(op(sphere), cell)(arg, more..., k) for
    k in axes(arg,1), cell in cells(sphere)
]

apply_trisk(op, sphere, arg::Vector, more...) = [
    op(op(sphere), edge)(arg, more...) for edge in edges(sphere)
]

apply_trisk(op, sphere, arg::Matrix, more...) = [
    op(op(sphere), edge)(arg, more..., k) for
    k in axes(arg,1), edge in edges(sphere)
]

apply(objects, op, sphere, arg::Vector, more...) = [op(op(sphere), obj)(arg, more...) for obj in objects(sphere)]
apply(objects, op, sphere, arg::Matrix, more...) = 
    [op(op(sphere), obj)(arg, more..., k) for k in axes(arg,1), obj in objects(sphere)]

# mesh objects
cells(sphere) = eachindex(sphere.xyz_i)
duals(sphere) = eachindex(sphere.xyz_v)
edges(sphere) = eachindex(sphere.xyz_e)

# operators
gradient(sphere, qi) = apply(edges, Stencils.gradient, sphere, qi)
gradperp(sphere, qv) = apply(edges, Stencils.gradperp, sphere, qv)
perp(sphere, un) = apply(edges, Stencils.perp, sphere, un)
centered_flux(sphere, qi, un) = apply(edges, Stencils.centered_flux, sphere, qi, un)
curl(sphere, ue) = apply(duals, Stencils.curl, sphere, ue)
average_iv(sphere, qi) = apply(duals, Stencils.average_iv_form, sphere, qi)
average_ie(sphere, qi) = apply(edges, Stencils.average_ie, sphere, qi)
average_ve(sphere, qv) = apply(edges, Stencils.average_ve, sphere, qv)
divergence(sphere, U) = apply(cells, Stencils.div_form, sphere, U)
gradient3d(sphere, U) = apply_primal(Stencils.gradient3d, sphere, U)
dot_product(sphere, U) = apply(cells, Stencils.dot_product_form, sphere, U, U)
squared_covector(sphere, U) = apply(cells, Stencils.squared_covector, sphere, U)
TRiSK(sphere, args...) = apply_trisk(Stencils.TRiSK, sphere, args...)

# error measures
Linf(x) = maximum(abs, x)
Linf(x, y) = maximum(abs(a - b) for (a, b) in zip(x, y))
maxeps(x) = Linf(x) * eps(eltype(x))

# compare scalars or 3D vectors (tuples of scalars)
approx3d(a::Tuple, b::Tuple; rtol, atol) =
    all(isapprox(x, y; rtol, atol) for (x, y) in zip(a, b))
approx3d(a, b; rtol, atol) = isapprox(a, b; rtol, atol)

maxabs(x::Number) = abs(x)
maxabs(x::Tuple) = maximum(maxabs, x)
maxabs(x::AbstractArray) = maximum(maxabs, x)

# check 3D operator
check_3D(op) = function(sphere, arg::Matrix, more...)
    ret = op(sphere, arg, more...)
    ref = op(sphere, arg[1,:], map(x->x[1,:], more)...)
    # @unroll uses muladd: fusion may differ between vector and matrix paths.
    # Compare at machine roundoff while retaining the optimized kernels.
    rtol = 10eps(eltype(arg))
    atol = rtol * maxabs(ref)
    @test all(approx3d(a, b; rtol, atol) for (a, b) in zip(ret[1,:], ref))
    return ret
end

function test_curlgrad(sphere, qi)
    gradq = check_3D(gradient)(sphere, qi)
    curlgradq = curl(sphere, gradq)
    @test Linf(curlgradq) < maxeps(gradq)
end

function test_divgradperp(sphere, psi)
    U = check_3D(gradperp)(sphere, psi)
    divU = check_3D(divergence)(sphere, U)
    @test Linf(divU) * maximum(sphere.Ai) < 2maxeps(U)
end

function test_TRiSK(sphere, phi, psi, qe)
    U = gradperp(sphere, psi) + gradient(sphere, phi)
    Uperp = check_3D(TRiSK)(sphere, U, qe)
    # Layers are independent. Summing identical roundoff errors over layers
    # grows linearly, whereas a sqrt(total length) bound grows sublinearly.
    for k in axes(U, 1)
        u, uperp = view(U, k, :), view(Uperp, k, :)
        sym = sum(a * b for (a, b) in zip(u, uperp))
        @test abs(sym) < 2Linf(uperp) * maxeps(u) * sqrt(length(u))
    end
end

function test_curlTRiSK(sphere, qi)
    U = check_3D(gradient)(sphere, qi) .* transpose(sphere.le_de) # cov => contra
    Uperp = TRiSK(sphere, U)
    curlUperp = check_3D(curl)(sphere, Uperp) ./ transpose(sphere.Av)
    # `divergence` returns a two-form, divide by the primal cell area to get a scalar;
    # `average_iv` returns the integral over the dual cell, divide by the dual area
    divU = check_3D(average_iv)(sphere, divergence(sphere, U) ./ transpose(sphere.Ai)) ./
           transpose(sphere.Av)
    @test Linf(curlUperp + divU) < 1e-12 # can we get closer to eps(Float64) ?
end

function test_perp(rtol, sphere, levels)
    gradz_n = [z for k in levels, (x, y, z) in sphere.normal_e] # ∇z, normal component
    gradz_t = [z for k in levels, (x, y, z) in sphere.tangent_e] # ∇z, tangential component
    @test Linf(check_3D(perp)(sphere, gradz_n), gradz_t) < Linf(gradz_n) * rtol
    # test dot_product with same data
    check_3D(dot_product)(sphere, gradz_n)
    check_3D(squared_covector)(sphere, gradz_n)
end

function test_div(rtol, sphere, levels)
    curlz = [z * le for k in levels, ((x, y, z), le) in zip(sphere.tangent_e, sphere.le)] # ∇z⟂, contravariant
    divcurlz = check_3D(divergence)(sphere, curlz)
    @test  Linf(divcurlz) < rtol
end

function test_gradient3d(rtol, sphere, qi)
    # q=sinϕ, |∇q|²=cos²ϕ  ⇒  |∇q|²+q²-1 = 0
    gradq = check_3D(gradient3d)(sphere, qi)
    check = (dotprod(gq, gq) + q^2 - 1 for (q, gq) in zip(qi, gradq))
    @test Linf(check) < rtol
end

function test_average(rtol, sphere, qi)
    qie = check_3D(average_ie)(sphere, qi)
    # average_iv_form integrates over a dual cell; average_ve takes a scalar.
    qiv = check_3D(average_iv)(sphere, qi) ./ transpose(sphere.Av)
    qve = check_3D(average_ve)(sphere, qiv)
    check_3D(centered_flux)(sphere, qi, qie)
    @test Linf(qie, qve) < 2rtol
end

dotprod(a::NTuple{3,F}, b::NTuple{3,F}) where {F} = @unroll sum(a[i] * b[i] for i = 1:3)
