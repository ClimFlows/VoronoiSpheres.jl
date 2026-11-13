module VoronoiSpheresAdaptExt

using VoronoiSpheres: VoronoiSphere
using VoronoiSpheres.Operators: DualFromPrimal, EdgeFromDual, Gradient, Divergence,
                                Curl, TRiSK, SquaredCovector, CenteredFlux,
                                CenteredTracerFlux, EnergyTRiSK, DivCenteredFlux, MulGradient
using Adapt: @adapt_structure 

@adapt_structure VoronoiSphere
@adapt_structure DualFromPrimal
@adapt_structure EdgeFromDual
@adapt_structure Gradient
@adapt_structure Divergence
@adapt_structure Curl
@adapt_structure TRiSK
@adapt_structure SquaredCovector
@adapt_structure CenteredFlux
@adapt_structure CenteredTracerFlux
@adapt_structure EnergyTRiSK
@adapt_structure DivCenteredFlux
@adapt_structure MulGradient

end
