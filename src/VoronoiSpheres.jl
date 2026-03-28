module VoronoiSpheres

using MutatingOrNot: void, Void
using ManagedLoops: @loops, @unroll, @with
using Random: MersenneTwister

using CFDomains: CFDomains, UnstructuredDomain, @fast, Shell, HyperDiffusion

include("julia/Zippers.jl") # for zipped broadcast
include("julia/voronoi_stencils.jl")
include("julia/voronoi_operators.jl")
include("julia/lazy_expressions.jl")
include("julia/VoronoiSphere.jl")

CFDomains.shell(nz, layer::VoronoiSphere) = Shell(nz, layer, VHLayout())

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end # module
