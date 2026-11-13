using Enzyme: Const as EConst, Duplicated, Reverse, autodiff
using CFDomains.LazyOperators: set!, setminus!, addto!, subfrom!
using Random: MersenneTwister

function test_enzyme_operator(op, out, inputs...; active=map(_ -> true, inputs))
    rng = MersenneTwister(65)
    seed = randn(rng, size(out))
    dirs = map(x -> randn(rng, size(x)), inputs)
    inputs_bar = map(x -> randn(rng, size(x)), inputs)
    initial_bars = map(copy, inputs_bar)
    arguments = map((x, dx, use) -> use ? Duplicated(x, dx) : EConst(x),
                    inputs, inputs_bar, active)
    result, result_bar = copy(out), copy(seed)
    autodiff(Reverse, Ops.apply!, EConst,
             Duplicated(result, result_bar), EConst(nothing), EConst(op), arguments...)

    expected = copy(out)
    Ops.apply!(expected, nothing, op, inputs...)
    @test result ≈ expected
    @test result_bar == (op.action! in (addto!, subfrom!) ? seed : zero(seed))
    for j in eachindex(inputs)
        active[j] || continue
        directional = ForwardDiff.derivative(0.0) do t
            perturbed = Base.setindex(inputs, inputs[j] .+ t .* dirs[j], j)
            target = convert.(typeof(t), out)
            Ops.apply!(target, nothing, op, perturbed...)
            dot(seed, target)
        end
        @test dot(inputs_bar[j] - initial_bars[j], dirs[j]) ≈ directional rtol=1e-10 atol=1e-10
    end
end

function overwrite_input!(out, op, q, u)
    Ops.apply!(out, nothing, op, q, u)
    fill!(q, 0)
    return nothing
end

@testset "Enzyme operator adjoints" begin
    mesh = VoronoiSphere(DYNAMICO_reader(ncread, VoronoiMeshes("uni.2deg.mesh.nc")); prec=Float64)
    rng = MersenneTwister(74)
    q = randn(rng, 3, length(mesh.lon_i))
    u = randn(rng, 3, length(mesh.lon_e))
    for action in (set!, setminus!, addto!, subfrom!)
        test_enzyme_operator(Ops.Divergence(mesh, action), q, u)
    end
    test_enzyme_operator(Ops.Gradient(mesh), u[1, :], q[1, :])
    test_enzyme_operator(Ops.SquaredCovector(mesh), q, u)
    test_enzyme_operator(Ops.CenteredFlux(mesh), u, q, u)
    test_enzyme_operator(Ops.CenteredFlux(mesh), u, q, u; active=(false, true))
    test_enzyme_operator(Ops.DivCenteredFlux(mesh, setminus!), q, q, u)
    test_enzyme_operator(Ops.EnergyTRiSK(mesh, addto!), u, u, randn(rng, size(u)))

    op = Ops.CenteredFlux(mesh)
    alias = Duplicated(copy(u), zero(u))
    @test_throws ArgumentError autodiff(Reverse, Ops.apply!, EConst,
        alias, EConst(nothing), EConst(op), EConst(q), alias)
    @test alias.val == u

    q0, u0 = copy(q), copy(u)
    dq, du, seed = zero(q), zero(u), randn(rng, size(u))
    autodiff(Reverse, overwrite_input!, EConst,
             Duplicated(zero(u), copy(seed)), EConst(op), Duplicated(q, dq), Duplicated(u, du))
    Ops.apply!(u, nothing, op, q0, u0)
    @test dot(du, u0) ≈ dot(seed, u)
end
