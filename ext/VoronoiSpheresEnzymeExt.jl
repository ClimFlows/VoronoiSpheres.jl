module VoronoiSpheresEnzymeExt

using Enzyme: Const, Duplicated, make_zero
using Enzyme.EnzymeRules: AugmentedReturn, RevConfigWidth, overwritten, runtime_activity
import Enzyme.EnzymeRules: augmented_primal, reverse
using VoronoiSpheres.Operators: VoronoiOperator, apply!, apply_internal!, apply_adj!

const Field = Union{Vector{<:AbstractFloat},Matrix{<:AbstractFloat}}
const Input = Union{Const{<:Field},Duplicated{<:Field}}

inactive(config, x::Duplicated) = runtime_activity(config) && x.val === x.dval
cotangent(config, x::Duplicated) = inactive(config, x) ? make_zero(x.val) : x.dval
cotangent(_, x::Const) = make_zero(x.val)

saved_input(x, overwritten) = overwritten ? copy(x) : x
saved_input(::Nothing, _) = nothing

function check_aliasing(output, input)
    Base.mightalias(output, input) &&
        throw(ArgumentError("Enzyme Voronoi adjoints require the output to be disjoint from the inputs"))
    return nothing
end

function augmented_primal(
    config::RevConfigWidth{1}, ::Const{typeof(apply!)}, ::Type{Const{Nothing}},
    output::Duplicated{<:Field}, mgr::Const, op::Const{<:VoronoiOperator{1,1}}, input::Input,
)
    check_aliasing(output.val, input.val)
    extra = apply_internal!(output.val, mgr.val, op.val, input.val)
    tape = saved_input(extra, overwritten(config)[5])
    return AugmentedReturn(nothing, nothing, tape)
end

function reverse(
    config::RevConfigWidth{1}, ::Const{typeof(apply!)}, ::Type{Const{Nothing}}, tape,
    output::Duplicated{<:Field}, mgr::Const, op::Const{<:VoronoiOperator{1,1}}, input::Input,
)
    if !inactive(config, output)
        apply_adj!(output.dval, mgr.val, op.val, cotangent(config, input), tape)
    end
    return (nothing, nothing, nothing, nothing)
end

function augmented_primal(
    config::RevConfigWidth{1}, ::Const{typeof(apply!)}, ::Type{Const{Nothing}},
    output::Duplicated{<:Field}, mgr::Const, op::Const{<:VoronoiOperator{1,2}},
    input1::Input, input2::Input,
)
    check_aliasing(output.val, input1.val)
    check_aliasing(output.val, input2.val)
    extras = apply_internal!(output.val, mgr.val, op.val, input1.val, input2.val)
    tape = map(saved_input, extras, overwritten(config)[5:6])
    return AugmentedReturn(nothing, nothing, tape)
end

function reverse(
    config::RevConfigWidth{1}, ::Const{typeof(apply!)}, ::Type{Const{Nothing}}, tape,
    output::Duplicated{<:Field}, mgr::Const, op::Const{<:VoronoiOperator{1,2}},
    input1::Input, input2::Input,
)
    if !inactive(config, output)
        apply_adj!(output.dval, mgr.val, op.val,
                   cotangent(config, input1), cotangent(config, input2), tape)
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

end
