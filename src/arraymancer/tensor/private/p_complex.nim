## Provides numeric identities, component casts and complex reduction arithmetic.
import std/atomics
from complex import Complex, complex32, complex64

template numberOne*(T:type SomeNumber):SomeNumber= T(1)
template numberOne*(T:type Complex[float32]):Complex[float32] = complex32(1.0, 0.0)
template numberOne*(T:type Complex[float64]):Complex[float64] = complex64(1.0, 0.0)

converter Complex64*[T:SomeNumber](x: T):Complex[float64]=
  result.re = x.float64
  result.im = 0'f64

converter Complex32*[T:SomeNumber](x: T):Complex[float32]=
  result.re = x.float32
  result.im = 0'f32


func convertedComplexReductionValue*[T; F: SomeFloat](value: T;
    component: typedesc[F]): Complex[F] {.inline.} =
  ## Keep casts in the scan loop and avoid double rounding integer components.
  when T is Complex:
    result.re = F(value.re)
    result.im = F(value.im)
  elif T is bool:
    result.re = F(ord(value))
  else:
    result.re = F(value)

var discardedImaginaryWarning: Atomic[bool]

proc warnComplexToRealCast*[T, U](source: typedesc[T]; dtype: typedesc[U];
                                 count: int) =
  ## Casting warns once across threads; boolean truth keeps both components.
  when T is Complex and U isnot Complex and U isnot bool:
    if count > 0 and not discardedImaginaryWarning.load(moRelaxed) and
        not discardedImaginaryWarning.exchange(true, moRelaxed):
      stderr.writeLine("Warning: Casting complex values to real discards the imaginary part")

