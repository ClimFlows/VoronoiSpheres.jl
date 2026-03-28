module VoronoiSpheresMooncakeExt

using Base: @propagate_inbounds as @prop
using CFDomains.LazyOperators: archive, restore!

import VoronoiSpheres.VoronoiOperators as Ops
using VoronoiSpheres.VoronoiOperators: apply!, apply_adj!, apply_internal!, VoronoiOperator

import Mooncake
using Mooncake: CoDual, NoTangent, NoPullback, NoFData, NoRData
using Mooncake: zero_fcodual, primal, tangent, lgetfield

Mooncake.tangent_type(::Type{<:VoronoiOperator}) = NoTangent

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoArray{F} = CoDual{<:AbstractArray{F}, <:AbstractArray{F}}
const CoNumber{F} = CoDual{F,NoFData}
const CoOperator{A,B} = CoDual{<:VoronoiOperator{A,B}, NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(apply!), Vararg}
Mooncake.rrule!!(::CoFunction(apply!), fx::Vararg) = apply!_rrule!!(fx...)

function apply!_rrule!!(foutput::CoArray{F}, fmgr::CoDual, op::CoOperator{1,1}, finput::CoArray{F}) where F
    # @info "apply!_rrule!!" typeof(foutput) typeof(op) typeof(finput)
    output, mgr, stencil, input = primal(foutput), primal(fmgr), primal(op), primal(finput)
    output0 = archive(output)    
    dout, din = tangent(foutput), tangent(finput)
    extras = apply_internal!(output, mgr, stencil, input) # inputs needed by pullback, if any
    function apply!_pullback!!(::NoRData)
        restore!(output, output0) # undo mutation
        apply_adj!(dout, mgr, stencil, din, extras)
        # rdata for (apply!, output, mgr, op, input)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), apply!_pullback!!
end

function apply!_rrule!!(foutput::CoArray{F}, fmgr::CoDual, op::CoOperator{1,2}, finput1::CoArray{F}, finput2::CoArray{F}) where F
#    @info "apply!_rrule!!" typeof(foutput) typeof(op) typeof(finput1) typeof(finput2)
    output, mgr, stencil, input1, input2 = map(primal, (foutput, fmgr, op, finput1, finput2))
    output0 = archive(output)
    ∂out, ∂in1,  ∂in2 = tangent(foutput), tangent(finput1), tangent(finput2)
    extras = apply_internal!(output, mgr, stencil, input1, input2) # inputs needed by pullback, if any
    function apply!_pullback!!(::NoRData)
        restore!(output, output0) # undo mutation
        apply_adj!(∂out, mgr, stencil, ∂in1, ∂in2, extras)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
    end
    return zero_fcodual(nothing), apply!_pullback!!
end

end
